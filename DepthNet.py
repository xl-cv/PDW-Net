import torch
import torch.nn as nn
import torch.nn.functional as F

class DRDB(nn.Module):
    def __init__(self, in_ch, growth_rate=32):
        super(DRDB, self).__init__()

        # 三个并行空洞卷积分支，感受野 d=1,2,3
        self.branch1 = nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=3, dilation=3)

        self.concat_ch = growth_rate * 3

        # 残差增强路径：5个 Dilated Conv + Residual
        self.enhance_blocks = nn.ModuleList()
        for _ in range(5):
            self.enhance_blocks.append(
                nn.Sequential(
                    nn.Conv2d(self.concat_ch, self.concat_ch, kernel_size=3, padding=2, dilation=2),
                    nn.ReLU(inplace=True)
                )
            )

        # 通道匹配：输入x映射到concat_ch
        self.match_conv = nn.Conv2d(in_ch, self.concat_ch, kernel_size=1)

        # 输出通道映射回in_ch，保证输出和输入通道数相同
        self.output_conv = nn.Conv2d(self.concat_ch, in_ch, kernel_size=1)

        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多尺度并行空洞卷积支路
        b1 = F.gelu(self.branch1(x))
        b2 = F.gelu(self.branch2(x))
        b3 = F.gelu(self.branch3(x))
        F_concat = torch.cat([b1, b2, b3], dim=1)  # [B, concat_ch, H, W]

        # 多层增强路径（×5），带残差连接
        out = F_concat
        for block in self.enhance_blocks:
            residual = out
            out = block(out)
            out = out + residual

        F_enhance = out

        # 输入特征 x 映射到 concat_ch
        x_resized = self.match_conv(x)

        # 三路融合输出 + 映射回输入通道
        out = x_resized + F_concat + F_enhance
        out = self.output_conv(out)

        # 最终激活
        out = self.final_activation(out)

        # 加残差连接，保持DRDB结构
        out = x + out

        return out


class DN(nn.Module):
    def __init__(self):
        super(DN, self).__init__()
        self.DRDB_layer1 = DRDB(in_ch=3, growth_rate=32)
        self.conv1 = nn.Conv2d(3, 24, 3, 2, 1)
        self.DRDB_layer2 = DRDB(in_ch=24, growth_rate=32)
        self.conv2 = nn.Conv2d(24, 48, 3, 2, 1)
        self.DRDB_layer3 = DRDB(in_ch=48, growth_rate=32)
        self.conv3 = nn.Conv2d(48, 96, 3, 2, 1)
        self.DRDB_layer4 = DRDB(in_ch=96, growth_rate=32)
        self.conv4 = nn.Conv2d(96, 128, 3, 2, 1)

        self.DRDB_layer5 = DRDB(in_ch=128, growth_rate=32)
        self.up1 = nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1)  # 上采样到[5, 96, 32, 32]
        self.DRDB_layer6 = DRDB(in_ch=96, growth_rate=32)
        self.up2 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1)  # 上采样到[5, 48, 64, 64]
        self.DRDB_layer7 = DRDB(in_ch=48, growth_rate=32)
        self.up3 = nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1)  # 上采样到[5, 24, 128, 128]
        self.DRDB_layer8 = DRDB(in_ch=24, growth_rate=32)
        self.up4 = nn.ConvTranspose2d(24, 3, kernel_size=4, stride=2, padding=1)  # 上采样到[5, 3, 256, 256]
        self.final_conv = nn.Conv2d(3,1,1)

    def forward(self, x):

        x1 = self.DRDB_layer1(x)
        x1 = self.conv1(x1)
        x1 = self.DRDB_layer2(x1)
        x1 = self.conv2(x1)
        x1 = self.DRDB_layer3(x1)
        x1 = self.conv3(x1)
        x1 = self.DRDB_layer4(x1)
        x1 = self.conv4(x1)
        x1 = self.DRDB_layer5(x1)
        x1 = self.up1(x1)
        x1 = self.DRDB_layer6(x1)
        x1 = self.up2(x1)
        x1 = self.DRDB_layer7(x1)
        x1 = self.up3(x1)
        x1 = self.DRDB_layer8(x1)
        x1 = self.up4(x1)
        x1 = self.final_conv(x1)

        return x1