# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Simple Baselines for Image Restoration.

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
"""

import torch
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from torch import nn


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=None, dec_blk_nums=None):
        if dec_blk_nums is None:
            dec_blk_nums = []
        if enc_blk_nums is None:
            enc_blk_nums = []
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.ending = nn.Conv2d(
            in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        _B, _C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        _N, _C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class Decoder(nn.Module):
    def __init__(
        self, width=32, use_indices=None, dec_blk_nums=None, dw_expand=2, ffn_expand=2, out_channels=3, target_size=2048
    ):
        """
        Args:
            width (int): 基础通道数 (Default 32).
            use_indices (list[int]): 选用的编码器层级索引 (0-4)。 必须从小到大排序。
            0: 1024x1024, 1: 512x512, ..., 4: 64x64.
            dec_blk_nums (list[int]): 每个选用层级对应的 NAFBlock 数量。
            dw_expand (int): NAFBlock 参数.
            ffn_expand (int): NAFBlock 参数.
            out_channels (int): 最终输出图像通道数 (Default 3).
            target_size (int): 期望的最终输出尺寸 (Default 2048).
        """
        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1, 1, 1]
        if use_indices is None:
            use_indices = [0, 1, 2, 3, 4]
        super().__init__()

        if len(use_indices) != len(dec_blk_nums):
            raise ValueError("Length of use_indices must match length of dec_blk_nums")

        self.use_indices = use_indices
        self.num_levels = len(use_indices)
        self.target_size = target_size

        # 固定的 5 层通道配置: [32, 64, 128, 256, 512]
        self.all_channels = [width * (2**i) for i in range(5)]

        # 获取选定层级的通道和尺寸信息
        self.selected_channels = [self.all_channels[i] for i in use_indices]

        # 计算选定中最浅层（Index最小）的分辨率，用于计算最终上采样倍数
        # Index 0 -> 1024, Index 1 -> 512, Index 2 -> 256 ...
        # 公式: Resolution = 2048 / (2^index)
        self.shallowest_index = use_indices[0]
        self.shallowest_resolution = 1024 // (2**self.shallowest_index)

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 构建模块 (从深层到浅层)
        for i in range(self.num_levels):
            rev_idx = self.num_levels - 1 - i  # 逆序: 0->Deep, N-1->Shallow
            current_chan = self.selected_channels[rev_idx]

            # 1. NAFBlock 处理单元
            decoder_layer = nn.Sequential(
                *[
                    NAFBlock(current_chan, DW_Expand=dw_expand, FFN_Expand=ffn_expand)
                    for _ in range(dec_blk_nums[rev_idx])
                ]
            )
            self.decoders.append(decoder_layer)

            # 2. 上采样模块 (如果不是最浅层)
            if rev_idx > 0:
                next_chan = self.selected_channels[rev_idx - 1]
                up_module = nn.Sequential(
                    nn.Conv2d(current_chan, next_chan * 4, kernel_size=1, bias=False), nn.PixelShuffle(2)
                )
                self.ups.append(up_module)

        # 3. 最终输出头
        # 将通道映射到 out_channels
        self.final_conv = nn.Conv2d(
            self.selected_channels[0], out_channels, kernel_size=3, padding=1, stride=1, bias=True
        )

        # 动态计算最后需要上采样多少倍才能回到 target_size (2048)
        # 例如：如果最浅用到 Index 2 (256x256)，则需要 2048/256 = 8倍上采样
        final_scale_factor = self.target_size / self.shallowest_resolution

        if final_scale_factor > 1:
            self.final_upsample = nn.Upsample(scale_factor=final_scale_factor, mode="bilinear", align_corners=False)
        else:
            self.final_upsample = nn.Identity()

    def forward(self, enc_skips_list):
        """
        Args:
            enc_skips_list (list[Tensor]): 编码器输出的 5 个阶段结果。
            顺序固定: [Stage_0, Stage_1, Stage_2, Stage_3, Stage_4].

        Returns:
            Tensor: [B, out_channels, 2048, 2048]
        """
        # 1. 提取选定的 Skips
        selected_skips = [enc_skips_list[i] for i in self.use_indices]

        # 2. 反转为从深到浅: [Deep, ..., Shallow]
        skips_reversed = selected_skips[::-1]

        # 3. 动态确定起始输入 (x_middle)
        # 起始输入就是选中列表里最深的那个特征图 (列表的第一个元素)
        out = skips_reversed[0]

        # 4. 逐层解码
        for i in range(self.num_levels):
            skip = skips_reversed[i]

            # A. 空间对齐
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)

            # B. 融合 (Add)
            out = out + skip

            # C. NAFBlock 处理
            out = self.decoders[i](out)

            # D. 上采样 (如果不是最后一层)
            if i < len(self.ups):
                out = self.ups[i](out)

        # 5. 最终输出处理
        # 此时 out 尺寸 = shallowest_resolution (例如 256, 512 或 1024)

        out = self.final_conv(out)  # torch.Size([3, 3, 1024, 1024])

        out = self.final_upsample(out)  # 变 2048x2048

        out = nn.Tanh()(out)
        return out


"""if __name__ == '__main__':
    img_channel = 3
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]


    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)"""
