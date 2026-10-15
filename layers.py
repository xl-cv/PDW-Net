#就是编码需要的

import torch
import torch.nn as nn
import torch.nn.functional as F
from dc import  *



'''class LightFrequencyAmplitudeModulation(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 保留原有全局可学习参数
        self.global_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.global_shift = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # 新增：局部感知的1×1卷积（参数极少，显存可忽略）
        self.local_avg_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
       

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 原有频域变换逻辑不变（保留核心频域增强能力）
        fft_x = torch.fft.rfft2(x, dim=(-2, -1))
        fft_amp = torch.abs(fft_x)  # 幅值：尺寸 [B, C, H, W//2 + 1]（如256→129）
        fft_phase = torch.angle(fft_x)  # 相位：尺寸与fft_amp一致

        # 2. 新增：提取局部空域信息（3×3局部窗口平均，模拟局部雾浓度，无额外显存开销）
        local_avg = F.avg_pool2d(x, kernel_size=3, padding=1, stride=1)  # 就地运算，不新增大张量
        local_avg = self.local_avg_conv(local_avg)
        local_scale = torch.sigmoid(local_avg)  # 转为0~1的局部调制系数，尺寸 [B, C, H, W]（原始256）

        # 3. 关键修复：将local_scale插值匹配fft_amp的尺寸（动态适配，无信息丢失）
        fft_h, fft_w = fft_amp.shape[-2], fft_amp.shape[-1]  # 获取频域张量的空间尺寸
        local_scale = F.interpolate(
            local_scale,
            size=(fft_h, fft_w),  # 匹配频域张量的高度和宽度
            mode='bilinear',      # 双线性插值，平滑过渡，不引入额外显存
            align_corners=False
        )

        # 4. 融合全局+局部的幅值调制（此时尺寸完全一致，可正常运算）
        fft_amp_mod = fft_amp * (self.global_scale + local_scale) + self.global_shift
        fft_amp_mod = torch.clamp(fft_amp_mod, min=self.eps)  # 防止除零

        # 5. 原有逆变换逻辑不变
        fft_x_mod = torch.complex(
            fft_amp_mod * torch.cos(fft_phase),
            fft_amp_mod * torch.sin(fft_phase)
        )
        x_freq = torch.fft.irfft2(fft_x_mod, s=x.shape[-2:], dim=(-2, -1))  # 还原为原始空域尺寸
        # 残差连接 + 异常值处理（保留原有兜底逻辑）
        x_out = torch.nan_to_num(x + x_freq, nan=0.0, posinf=1e3, neginf=-1e3)
        return x_out

# 第二步：保留你原有 LightFrequencyChannelAttention 逻辑不变
class LightFrequencyChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        mid_ch = max(1, channels // reduction)
        # 通道注意力生成器
        self.fc1 = nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fft_x = torch.fft.rfft2(x, dim=(-2, -1))
        fft_amp = torch.abs(fft_x)
        # 频域全局池化：提取通道级统计信息
        freq_pool = torch.mean(fft_amp, dim=(-2, -1), keepdim=True)

        # 生成通道注意力权重
        attn = self.fc1(freq_pool)
        attn = F.relu(attn, inplace=True)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)

        # 幅值加权 + 逆变换
        fft_amp_weighted = fft_amp * attn
        fft_phase = torch.angle(fft_x)
        fft_x_mod = torch.complex(
            fft_amp_weighted * torch.cos(fft_phase),
            fft_amp_weighted * torch.sin(fft_phase)
        )
        x_freq = torch.fft.irfft2(fft_x_mod, s=x.shape[-2:], dim=(-2, -1))
        x_out = torch.nan_to_num(x + x_freq, nan=0.0, posinf=1e3, neginf=-1e3)
        return x_out


# 第三步：新增轻量化空间门控（解决高频良莠不分，显存开销＜100MB）
class LightSpatialGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1×1卷积降维/升维 + 深度卷积（极致轻量化，参数量极少）
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)  # 降维减少参数
        self.conv_dw = nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1, groups=channels // 4, bias=False)  # 深度卷积，显存友好
        self.conv2 = nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False)  # 升维还原通道
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_ori):
        # x：频域增强输出；x_ori：原始输入（提取空间结构信息，无额外显存开销）
        residual = x - x_ori  # 仅提取增强部分的残差，不新增大张量
        # 生成空间掩码（区分有用高频区域和噪声区域）
        gate = self.conv1(residual)
        gate = F.relu(gate, inplace=True)
        gate = self.conv_dw(gate)
        gate = F.relu(gate, inplace=True)
        gate = self.conv2(gate)
        gate = self.sigmoid(gate)
        # 门控加权：有用区域保留增强，噪声区域削弱增强
        x_out = x_ori + residual * gate
        return x_out

# 第四步：整合所有模块，形成最终优化版 LightFreqEnhanceBlock
class OptimizedLightFreqEnhanceBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 保留你原有核心模块
        self.lfam = LightFrequencyAmplitudeModulation(channels)
        self.lfca = LightFrequencyChannelAttention(channels, reduction)
        # 新增轻量化空间门控（仅此处新增，显存可忽略）
        self.spatial_gate = LightSpatialGate(channels)

    def forward(self, x):
        x_ori = x  # 保存原始输入，用于空间门控计算
        # 原有频域增强逻辑不变
        x = self.lfam(x)
        x = self.lfca(x)
        # 新增空间门控，筛选有用高频，抑制噪声（无额外显存压力）
        x = self.spatial_gate(x, x_ori)
        return x'''





######################LFAM频率增强#########
'''class LFAM_AttInv_Enhanced(nn.Module):
    """去雾专用终极版（显存充足+无分组卷积+快速收敛+高性能）：
    核心优化：
    1. 移除多尺度池化/正交初始化：解决前期收敛慢，梯度集中更新核心参数
    2. 幅值低频：5×5池化 + 5×5卷积（无通道缩减，特征无损）
    3. 幅值高频：7×7普通卷积（无分组，跨通道学习，细节恢复更连贯）
    4. 相位增强：3×3独立卷积+L2归一化+GELU（数值安全，避免噪声）
    5. 全链路数值安全：eps防除0、ReLU截断负值、tanh限制相位范围
    """
    def __init__(self, channels, eps=1e-6, negative_slope=0.1):
        super().__init__()
        self.eps = eps          # 仅用于数值安全，避免除0
        self.channels = channels
        self.negative_slope = negative_slope

        # ========== 幅值低频提取（5×5卷积+无通道缩减，特征无损） ==========
        self.freq_pool_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        )

        # ========== 相位cos/sin独立卷积（3×3分组保留：相位需通道独立） ==========
        # 相位卷积保留分组：相位是通道独立的结构信息，跨通道学习会引入噪声
        self.phase_cos_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.phase_sin_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

        # ========== 幅值高频增强（7×7普通卷积，无分组，跨通道学习） ==========
        # 关键修改：移除groups=channels，改为普通卷积
        self.amp_high_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, bias=False),  # 移除groups参数
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # ========== 相位高频增强（1×1+GELU，平滑降噪） ==========
        self.phase_high_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),  # 比LeakyReLU更平滑，适配相位高频增强
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"输入通道数{C}与模块通道数{self.channels}不匹配"

        # 1. 频域变换（基础操作）
        fft_x = torch.fft.rfft2(x, dim=(-2, -1))
        fft_amp = torch.abs(fft_x)       # 幅值（对应图像强度/细节）
        fft_phase = torch.angle(fft_x)   # 相位（对应图像结构/轮廓）

        # 2. 幅值增强（核心：低频提取+高频增强）
        # 2.1 5×5池化提取幅值低频（匹配5×5卷积感受野）
        fft_amp_pooled = F.avg_pool2d(fft_amp, kernel_size=5, stride=1, padding=2)
        # 2.2 5×5卷积学习低频特征（无通道缩减，特征无损）
        amp_low_conv = self.freq_pool_conv(fft_amp_pooled)
        # 2.3 高频分量分离+截断负值（数值安全）
        fft_amp_high = fft_amp - amp_low_conv
        fft_amp_high = torch.relu(fft_amp_high)  # 避免负值干扰增强效果
        # 2.4 7×7普通卷积增强高频（跨通道学习，细节更连贯）
        amp_high_att = self.amp_high_enhance(fft_amp_high)
        fft_amp_high_mod = fft_amp_high * amp_high_att
        # 2.5 低频+增强高频融合（最终幅值）
        fft_amp_final = amp_low_conv + fft_amp_high_mod

        # 3. 相位增强（核心：低频提取+平滑高频增强）
        # 3.1 相位转cos/sin（单位圆映射，数值安全）
        phase_cos = torch.cos(fft_phase)
        phase_sin = torch.sin(fft_phase)
        # 3.2 3×3独立卷积提取相位低频（保留分组，避免噪声）
        cos_low = self.phase_cos_conv(phase_cos)
        sin_low = self.phase_sin_conv(phase_sin)
        # 3.3 L2归一化（防除0，保证atan2稳定性）
        norm = torch.sqrt(cos_low**2 + sin_low**2 + self.eps)
        cos_low = cos_low / norm
        sin_low = sin_low / norm
        # 3.4 逆映射回相位（数值绝对安全）
        phase_low = torch.atan2(sin_low, cos_low)
        # 3.5 高频分量分离+GELU平滑增强
        phase_high = fft_phase - phase_low
        phase_high_att = self.phase_high_enhance(phase_high)
        phase_high_enhanced = torch.tanh(phase_high) * phase_high_att  # 限制相位范围，避免溢出
        # 3.6 低频+增强高频融合（最终相位）
        fft_phase_enhanced = phase_low + phase_high_enhanced

        # 4. 频域逆变换（还原空域图像）
        fft_x_final = torch.complex(
            fft_amp_final * torch.cos(fft_phase_enhanced),
            fft_amp_final * torch.sin(fft_phase_enhanced)
        )
        x_out = torch.fft.irfft2(fft_x_final, s=(H, W), dim=(-2, -1))

        return x_out'''
class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y)
        return y.view(B, C, 1, 1)


####################频率增强模块#####################
class ISFCBlock(nn.Module):
    """
    Interactive Spatial-Frequency Calibration (ISFC) Block (Clean & Direct)
    - 上分支：频域 2D FFT 解耦幅值与相位，对数能量通道重标定
    - 下分支：空域 1x1 + 3x3 DWConv + 1x1 结构门控
    - 交互：零中心频域残差与空域门控逐元素相乘，直接加回原特征
    """
    def __init__(self, channels: int = 24, reduction: int = 8):
        super().__init__()
        self.channels = channels

        # ================= 1. 频域分支 =================
        self.global_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.global_shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

        mid_ch = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # ================= 2. 空域门控分支 =================
        gate_ch = max(1, channels // 4)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, gate_ch, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(gate_ch, gate_ch, kernel_size=3, padding=1, groups=gate_ch, bias=True),
            nn.GELU(),
            nn.Conv2d(gate_ch, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # -------------- 频域分支 --------------
        fft_x = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        amp = torch.abs(fft_x) + 1e-8
        phase = torch.angle(fft_x)

        # 对数池化兼顾高低频能量
        log_amp = torch.log1p(amp)
        gap = torch.mean(log_amp, dim=(-2, -1), keepdim=True)
        w = self.mlp(gap)

        # 频域幅值调制与逆变换
        amp_mod = amp * w * self.global_scale + self.global_shift
        amp_mod = torch.clamp(amp_mod, min=1e-6)

        fft_mod = torch.polar(amp_mod, phase)
        x_freq = torch.fft.irfft2(fft_mod, s=(H, W), dim=(-2, -1), norm="ortho")
        
        # 提取频域净增量残差
        delta_freq = x_freq - x

        # -------------- 空域门控与直接融合 --------------
        gate = self.spatial_gate(x)
        out = x + delta_freq * gate

        return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

      #  Efficient up-convolution block (EUCB)两级上采样
class EUCB2x(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB2x, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class EUCB_4x(nn.Module):           #四级上采样  
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB_4x, self).__init__()
        self.in_channels = in_channels
        # 中间通道数：第一次上采样后用输入通道的1/2，平衡计算量
        mid_channels = in_channels // 2 if in_channels >= 2 else 1
        
        # 第一级2倍上采样（输出2H×2W）
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size//2, groups=in_channels, bias=False),  # 深度卷积
            nn.BatchNorm2d(in_channels),
            act_layer(activation, inplace=True)
        )
        
        # 通道调整+混洗（连接两级上采样）
        self.shuffle1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        
        # 第二级2倍上采样（输出4H×4W）
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size//2, groups=mid_channels, bias=False),  # 深度卷积
            nn.BatchNorm2d(mid_channels),
            act_layer(activation, inplace=True)
        )
        
        # 最终通道调整+混洗
        self.shuffle2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        # 第一级：2倍上采样
        x = self.up1(x)  # (B, in_channels, 2H, 2W)
        x = channel_shuffle(x, groups=self.in_channels)  # 通道混洗
        x = self.shuffle1(x)  # 调整通道数到mid_channels
        
        # 第二级：再2倍上采样（总4倍）
        x = self.up2(x)  # (B, mid_channels, 4H, 4W)
        x = channel_shuffle(x, groups=x.size(1))  # 基于当前通道数混洗
        x = self.shuffle2(x)  # 调整到目标通道数
        
        return x

#############先通道后空间#############



class MultiScaleGuidedResidualFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.fft = ISFCBlock(channels)   #频域增强
        self.se=SEAttention(3*channels)   #通道筛选
        self.compress_channels = (3 * channels) // 2
        self.conv1x1_half_compress = nn.Conv2d(
            in_channels=3 * channels,
            out_channels=self.compress_channels,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.rconv = nn.Conv2d(
            in_channels=self.compress_channels,
            out_channels=channels,  # 先默认映射到模块核心channels，后续动态对齐
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
       
        #self.iwp_dir_2x = nn.AvgPool2d(kernel_size=2, stride=2)
        #self.iwp_dir_4x = nn.AvgPool2d(kernel_size=4, stride=4)
        self.iwp_dir_2x = IWP_Dir(channels)   #变为原来的1/2  小波池化 模拟两倍下采样
        self.iwp_dir_4x=IWP_Dir_1_4_Light(channels) #变为原来的1/4  小波池化  模拟四倍下采样
        #self.kj=SpatialAttentionBlock(3 * channels)

        self.eucb_2x = EUCB2x(
            in_channels=channels,
            out_channels=channels
        )  
        self.eucb_4x=EUCB_4x(
            in_channels=channels,
            out_channels=channels
        )
        # 方向小波卷积（你已导入WTConv2d_Dir，无需重新定义）   换成7成7卷积
        '''
        self.conv = nn.Conv2d(in_channels=channels,, out_channels=channels,  kernel_size=7, stride=1, padding=3, bias=True)'''
        self.wt_conv_origin = WTConv2d_Dir(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,                     #7
            wt_type='db1'
        )                             
        self.wt_conv_1_2 = WTConv2d_Dir(
            in_channels=channels,
            out_channels=channels,         #5
            kernel_size=7,
            wt_type='db1'
        )
        self.wt_conv_1_4 = WTConv2d_Dir(        
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,              #3
            wt_type='db1'
        )
    def forward(self, x):
        n, c, h, w = x.size()
        xorigin=self.conv(x)
        xwt_ori=self.wt_conv_origin(xorigin)   #
        xwt_ori_fft=self.fft(xwt_ori)    #频域增强原尺度

        # H/4分支
        x1=self.iwp_dir_4x(x)
        x11=self.wt_conv_1_4(x1)
        x11ft=self.fft(x11)
        
        x12=self.eucb_2x(x11ft)
        x1_final=self.eucb_4x(x11ft)
      
        # H/2分支 
        x2=self.iwp_dir_2x(x)
        x2_1=x12+x2
        x2_11=self.wt_conv_1_2(x2_1)
        x2_11fft=self.fft(x2_11)
        
        x2_final=self.eucb_2x(x2_11fft)

        concat_feats = torch.cat([x1_final, x2_final, xwt_ori_fft], dim=1)  #72通道
        concat_s=self.se(concat_feats) #筛选72通道的有用信息
        concat_se= concat_s * concat_feats  #通道注意力加权,筛选有用通道数
        #concat_kj=self.kj(concat_se)  #局部空间注意力  先通道后空间
        concat_half=self.conv1x1_half_compress(concat_se)#36通道
        concat_final=self.rconv(concat_half)  #现在是24通道
        output=concat_final+x
        return output

# ---------------------- 基础模块（修改ResBlock核心） ----------------------
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# 核心修改：ResBlock中替换DeepPoolLayer为MultiScaleGuidedResidualFusion
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel,filter=False):  # 删掉data、filter参数
        super(ResBlock, self).__init__()
        # 保留通道一致性检查（软约束更友好，避免直接报错）
        if in_channel != out_channel:
            print(f"警告：ResBlock输入通道{in_channel}≠输出通道{out_channel}，已强制对齐为{in_channel}")
            out_channel = in_channel  # 强制通道一致，适配你的模块
        
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            MultiScaleGuidedResidualFusion(channels=in_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x




























'''class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, data, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel, data) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, data):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]

        if data == 'ITS' or 'Densehaze' or 'Haze4k' or 'Ihaze' or 'Nhhaze' or 'NHR' or 'Ohaze':
            dilation = [7,9,11]
        elif data == 'GTA5':
            dilation = [5,9,11]
            
        pools, convs, dynas = [],[],[]
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl


class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()
    
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(self.dilation*(kernel_size-1)//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None,:,None,None]

        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.) 

        return out_low + out_high


class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation*(kernel-1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None,1) if H else (1, None) 
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)

        return out_low + out_high


class MultiShapeKernel(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):

        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1+x2'''