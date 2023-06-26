from typing import Dict
from src.init_weights import init_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class AMS(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        auto_padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=auto_padding, groups=dim, dilation=dilation)


        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv(x)

        attna = self.conv0(attn)

        attn_0 = self.conv0_1(attna)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attna)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attna)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attna + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class Attention(nn.Module):
    def __init__(self, d_model, dilation=1):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(d_model, d_model, 1)
        self.act = nn.GELU()
        self.ams = AMS(d_model, dilation=dilation)
        self.conv2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.ams(x)
        x = self.conv2(x)
        return x

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = d_model * 4
        self.conv1 = nn.Conv2d(d_model, hidden_dim, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(hidden_dim, d_model, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x

class AMSE(nn.Module):
    def __init__(self, dim, dilation=1):
        super(AMSE, self).__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(dim)
        self.ffn = FFN(dim)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.bn1(x)
        x = self.attn(x)
        x = self.bn2(x)
        x = self.ffn(x)
        x = x + shortcut
        x = self.act(x)
        return x

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, layer_num=1, dilation=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(AMSE(out_channels, dilation=dilation))
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers
        )

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.squeeze = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dim // reduction, dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_num=1):
        super(Up, self).__init__()
        C = in_channels // 2
        self.norm = nn.BatchNorm2d(C)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(AMSE(out_channels))
        self.conv = nn.Sequential(*layers)
        self.satt = SpatialAttention(C)
        self.catt = ChannelAttention(C, reduction=4)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.norm(x1)
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        satt = self.satt(x2)
        catt = self.catt(x2)
        x2 = x2 + (satt * catt)
        x = self.conv1x1(torch.cat([x2, x1], dim=1))
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class AMSUnet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear=True,
                 base_c: int = 32):
        super(AMSUnet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(base_c),
            nn.GELU(),
            AMSE(base_c)
        )
        self.down1 = Down(base_c, base_c * 2, dilation=1)
        self.down2 = Down(base_c * 2, base_c * 4, dilation=1)
        self.down3 = Down(base_c * 4, base_c * 8, dilation=3)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16//factor, dilation=5)

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return {"out": logits}


if __name__ == '__main__':
    model = AMSUnet(in_channels=3, num_classes=2, base_c=32).to('cpu')
    input = torch.randn(1, 3, 480, 480).to('cpu')
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')