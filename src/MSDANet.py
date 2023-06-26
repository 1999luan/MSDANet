from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
from src.init_weights import init_weights
from collections import OrderedDict

# 定义MLP模块
class MLP(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#定义PSP模块
class PSPModule(nn.Module):
    def __init__(self, features, mid_features=1024, out_features=32, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), mid_features, kernel_size=1)
        self.out_conv = nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1, bias=False) #新加入卷积层（也可以将中间卷积作为最终卷积）
        self.out_bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        out_conv = self.out_conv(bottle)
        out = self.out_bn(out_conv)
        return self.relu(out)

#定义改编SK
class SKAttention(nn.Module):

    def __init__(self, channel=32, kernels=[3, 5, 7], reduction=4, L=32, base_atrous_rate=[4, 6, 8]):
        super().__init__()
        self.d = max(L, channel//reduction)
        self.convs = nn.ModuleList([])
        self.dconvs = nn.ModuleList([])
        i = 0
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k//2, bias=False)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
            self.dconvs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=3, padding=base_atrous_rate[i], dilation=base_atrous_rate[i], bias=False)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
            i = i+1
        self.advg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)
        self.ln = nn.LayerNorm(channel)
        self.mlp = MLP(in_features=channel, hidden_features=channel * 4)
        self.cat_conv = Cat_conv(2 * channel, channel)
        self.fc = nn.Linear(channel, self.d)
        self.gelu = nn.GELU()
        self.fcs1 = nn.ModuleList([])
        self.fcs2 = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs1.append(nn.Linear(self.d, channel))

        for i in range(2):
            self.fcs2.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def sk_sum(self, conv_outs, bs, c, flag):
        feats = torch.stack(conv_outs, 0)
        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = self.advg_pool(U).view(bs, c)
        S = self.ln(S)
        S = self.mlp(S)
        Z = self.fc(S)  # bs,d
        Z = self.gelu(Z)
        Z = self.drop(Z)

        ### calculate attention weight
        weights = []
        if flag:
            for fc in self.fcs1:
                weight = fc(Z)
                weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        else:
            for fc in self.fcs2:
                weight = fc(Z)
                weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V

    def sk_process(self, x, convs, flag=True):
        bs, c, _, _ = x.size()
        conv_outs1 = []
        ### split
        for conv in convs:
            conv_outs1.append(conv(x))
        V = self.sk_sum(conv_outs1, bs, c, flag)
        return V


    def forward(self, x):
        bs, c, _, _ = x.size()
        V1 = self.sk_process(x, self.convs)
        V2 = self.sk_process(x, self.dconvs)
        # conv_outs2 = [V1, V2]
        # V = self.sk_sum(conv_outs2, bs, c, flag=False)
        V = self.cat_conv(V1, V2)
        V = V + x
        return V


#定义SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
        self.ln = nn.LayerNorm(channel)
        self.mlp = MLP(in_features=channel, hidden_features=channel*4)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y1 = self.ln(y)
        # y = self.mlp(y1) + y
        y = self.ln(y)
        y = self.mlp(y)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)  #不改变特征层的大小 input(2, 512, 480, 480) --> output(2, 512, 480, 480)

#定义Inception V2模块
#构建基础的卷积模块，与Inception v1的基础模块相比，增加了BN层
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x = self.relu(x1 + x2)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels=None, atrous_rate=8, base_atrous_rate=[4, 6, 8], flag=False):
        super(Down, self).__init__()
        if out_channels is None:
            out_channels = 2*in_channels
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.advg_pool = nn.AdaptiveAvgPool2d(1)
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        if flag:
            self.cat_convs = Cat_conv(2 * in_channels, in_channels)
            self.sk = SKAttention(in_channels, base_atrous_rate=base_atrous_rate)
        else:
            self.cat_convs = Cat_conv(2 * in_channels, 2 * in_channels)
            self.sk = SKAttention(2 * in_channels, base_atrous_rate=base_atrous_rate)

    def forward(self, x):
        b, c, _, _ = x.size()
        x_max = self.max_pool(x)
        x_conv1 = self.convs1(x)
        x_conv2 = self.convs2(x_conv1)
        x = self.cat_convs(x_max, x_conv2)
        x = self.sk(x)
        return x

class Cat_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cat_conv, self).__init__()
        self.conv = Conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.se = SELayer(out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class In_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(In_Conv, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, kernel_size=1),
             nn.BatchNorm2d(out_channels),
         )
        self.gelu = nn.GELU()
    def forward(self, x):
        x1 = self.in_conv(x)
        return x1

def Calculate_atrous_rate(image_size, i):
    d1 = math.floor(math.sqrt(image_size))-(4 + i)
    d2 = math.floor(math.sqrt(image_size)) - (2 + i)
    d3 = math.floor(math.sqrt(image_size)) - i
    factor = [d1, d2, d3]
    return factor

class MSDANet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 image_size: int = 240,
                 base_c: int = 64):
        super(MSDANet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = In_Conv(in_channels, base_c)
        self.cat_conv0 = Cat_conv(2*base_c, base_c)
        self.psp0 = PSPModule(features=base_c, out_features=32)

        self.down1 = Down(base_c, atrous_rate=6, base_atrous_rate=Calculate_atrous_rate(image_size//2, 2))
        self.cat_conv1 = Cat_conv(4 * base_c, 2 * base_c)
        self.psp1 = PSPModule(features=64, out_features=64)

        self.down2 = Down(base_c * 2, atrous_rate=5, base_atrous_rate=Calculate_atrous_rate(image_size//2, 3))
        self.cat_conv2 = Cat_conv(8 * base_c, 4 * base_c)
        self.psp2 = PSPModule(features=128, out_features=128)

        self.down3 = Down(base_c * 4, atrous_rate=4, base_atrous_rate=Calculate_atrous_rate(image_size//2, 4))
        self.cat_conv3 = Cat_conv(16 * base_c, 8 * base_c)
        self.psp3 = PSPModule(features=256, out_features=256)

        factor = 2 if bilinear else 1

        self.down4 = Down(base_c * 8, out_channels=base_c * 8, atrous_rate=3, base_atrous_rate=Calculate_atrous_rate(image_size//2, 5), flag=True)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x1 = self.in_conv(x)
        #x1 = self.sk0(x1)
        x1_1 = self.psp0(x1)
        x1_1 = self.cat_conv0(x1, x1_1)
        x2 = self.down1(x1)
        x2_1 = self.psp1(x2)
        x2_1 = self.cat_conv1(x2, x2_1)
        x3 = self.down2(x2)
        x3_1 = self.psp2(x3)
        x3_1 = self.cat_conv2(x3, x3_1)
        x4 = self.down3(x3)
        x4_1 = self.psp3(x4)
        x4_1 = self.cat_conv3(x4, x4_1)
        x5 = self.down4(x4)
        x = self.up1(x5, x4_1)
        x = self.up2(x, x3_1)
        x = self.up3(x, x2_1)
        x = self.up4(x, x1_1)
        logits = self.out_conv(x)

        return {"out": logits}


if __name__ == '__main__':
    if __name__ == "__main__":
        input = torch.randn(2, 3, 480, 480)
        model = MSDANet(in_channels=3, num_classes=2, base_c=32)
        print(input.shape)
        output = model(input)
        print(output['out'].shape)

        from ptflops import get_model_complexity_info

        flops, params = get_model_complexity_info(model, input_res=(3, 240, 240), as_strings=True,
                                                  print_per_layer_stat=False)
        print('      - Flops:  ' + flops)
        print('      - Params: ' + params)
    # input = torch.randn(2, 3, 240, 240)
    # print(input.shape)
    # model = MSDANet(in_channels=3, num_classes=2, base_c=32)
    # output = model(input)
    # #torch.sigmoid()
    # #print(output.shape)
    # print(output['out'].shape)
    # # model = LZ_UNet(in_channels=3, num_classes=2, base_c=32)
    # # if torch.cuda.is_available():
    # #     model.cuda()
    # # summary(model, input_size=(3, 480, 480), batch_size=1)
