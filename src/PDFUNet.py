import torch
import torch.nn as nn
from torchsummary import summary
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1):
        super(ConvLayer, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, r):
        super(SEBlock, self).__init__()

        redu_chns = int(in_channels / r)
        self.se_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, redu_chns, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(redu_chns, in_channels, kernel_size=1, padding=0),
            nn.ReLU())

    def forward(self, x):
        f = self.se_layers(x)
        return f*x + x

class PDFBlock(nn.Module):
    def __init__(self,in_channels, out_channels_list, kernel_size_list, dilation_list):
        super(PDFBlock, self).__init__()
        self.conv_num = len(out_channels_list)
        assert(self.conv_num == 4)
        assert(self.conv_num == len(kernel_size_list) and self.conv_num == len(dilation_list))
        pad0 = int((kernel_size_list[0] - 1) / 2 * dilation_list[0])
        pad1 = int((kernel_size_list[1] - 1) / 2 * dilation_list[1])
        pad2 = int((kernel_size_list[2] - 1) / 2 * dilation_list[2])
        pad3 = int((kernel_size_list[3] - 1) / 2 * dilation_list[3])
        self.conv_1 = nn.Conv2d(in_channels, out_channels_list[0], kernel_size = kernel_size_list[0], dilation = dilation_list[0], padding = pad0 )
        self.conv_2 = nn.Conv2d(in_channels, out_channels_list[1], kernel_size = kernel_size_list[1], dilation = dilation_list[1], padding = pad1 )
        self.conv_3 = nn.Conv2d(in_channels, out_channels_list[2], kernel_size = kernel_size_list[2], dilation = dilation_list[2], padding = pad2 )
        self.conv_4 = nn.Conv2d(in_channels, out_channels_list[3], kernel_size = kernel_size_list[3], dilation = dilation_list[3], padding = pad3 )

        out_channels  = out_channels_list[0] + out_channels_list[1] + out_channels_list[2] + out_channels_list[3]
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        y  = torch.cat([x1, x2, x3, x4], dim=1)
        y  = self.conv_1x1(y)
        return y

class ConBNActBlock(nn.Module):
    def __init__(self,in_channels, out_channels, dropout_p):
        super(ConBNActBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            SEBlock(out_channels, 2)
        )

    def forward(self, x):
        return self.conv_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels,
                 bilinear=True, dropout_p = 0.5):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConBNActBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1    = self.up(x1)
        x_cat = torch.cat([x2, x1], dim=1)
        y     = self.conv(x_cat)
        return y + x_cat

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.conv    = ConBNActBlock(2 * in_channels, out_channels, dropout_p)

    def forward(self, x):
        x_max = self.maxpool(x)
        x_avg = self.avgpool(x)
        x_cat = torch.cat([x_max, x_avg], dim=1)
        y     = self.conv(x_cat)
        return y + x_cat

class PDFUNet (nn.Module):
    def __init__(self, in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True):
        super(PDFUNet, self).__init__()
        self.in_channels = in_channels
        self.f_chan = [32, 64, 128, 256, 512]
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.dropout = [0.0, 0.0, 0.3, 0.4, 0.5]
        assert(len(self.f_chan) == 5)

        f0_half = int(self.f_chan[0] / 2)
        f1_half = int(self.f_chan[1] / 2)
        f2_half = int(self.f_chan[2] / 2)
        f3_half = int(self.f_chan[3] / 2)
        self.in_conv= ConBNActBlock(self.in_channels, self.f_chan[0], self.dropout[0])
        self.down1  = DownBlock(self.f_chan[0], self.f_chan[1], self.dropout[1])
        self.down2  = DownBlock(self.f_chan[1], self.f_chan[2], self.dropout[2])
        self.down3  = DownBlock(self.f_chan[2], self.f_chan[3], self.dropout[3])
        self.down4  = DownBlock(self.f_chan[3], self.f_chan[4], self.dropout[4])

        self.bridge0= ConvLayer(self.f_chan[0], f0_half)
        self.bridge1= ConvLayer(self.f_chan[1], f1_half)
        self.bridge2= ConvLayer(self.f_chan[2], f2_half)
        self.bridge3= ConvLayer(self.f_chan[3], f3_half)

        self.up1    = UpBlock(self.f_chan[4], f3_half, self.f_chan[3], dropout_p = self.dropout[3])
        self.up2    = UpBlock(self.f_chan[3], f2_half, self.f_chan[2], dropout_p = self.dropout[2])
        self.up3    = UpBlock(self.f_chan[2], f1_half, self.f_chan[1], dropout_p = self.dropout[1])
        self.up4    = UpBlock(self.f_chan[1], f0_half, self.f_chan[0], dropout_p = self.dropout[0])

        f4 = self.f_chan[4]
        aspp_chns = [int(f4 / 4), int(f4 / 4), int(f4 / 4), int(f4 / 4)]
        aspp_knls = [1, 3, 3, 3]
        aspp_dila = [1, 2, 4, 6]
        self.aspp = PDFBlock(f4, aspp_chns, aspp_knls, aspp_dila)


        self.out_conv = nn.Conv2d(self.f_chan[0], self.num_classes,
            kernel_size = 3, padding = 1)

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)
        x0  = self.in_conv(x)
        x0b = self.bridge0(x0)
        x1  = self.down1(x0)
        x1b = self.bridge1(x1)
        x2  = self.down2(x1)
        x2b = self.bridge2(x2)
        x3  = self.down3(x2)
        x3b = self.bridge3(x3)
        x4  = self.down4(x3)
        x4  = self.aspp(x4)

        x   = self.up1(x4, x3b)
        x   = self.up2(x, x2b)
        x   = self.up3(x, x1b)
        x   = self.up4(x, x0b)
        output = self.out_conv(x)

        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return {"out": output}

if __name__ == '__main__':
    input = torch.randn(2, 3, 480, 480)
    model = PDFUNet()
    print(input.shape)
    output = model(input)
    print(output['out'].shape)
    # if torch.cuda.is_available():
    #     model.cuda()
    # summary(model, input_size=(3, 240, 240), batch_size=1)
