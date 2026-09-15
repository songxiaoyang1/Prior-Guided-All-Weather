import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.basicsr.models.archs.NAFNet_arch import NAFBlock,SimpleGate  


class SimpleGateOld(nn.Module):
    """NAFNet 中的简单门控机制"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlockOld(nn.Module):
    """
    轻量级 NAF Block，用于图像复原特征提取。
    去除了复杂的频域操作，保留核心的 Spatial Gating 和 Channel Attention。
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, bias=True),
        )
        
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, bias=True)

        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma

class RestoreBranchOld(nn.Module):
    """
    并行图像复原分支。
    """
    def __init__(self, *args):
        """
        Args:
            支持框架的参数传递格式
        """
        super().__init__()
        
        # 初始化默认值
        in_channels_list = [64]
        mid_channels = 64
        num_naf_blocks = 4
        upsample_factor = 1
        
        # 解析参数
        if len(args) >= 7:
            # 框架传递了完整的参数：in_channels_list, mid_ch, naf_blocks, upsample_factor, reg_max, end2end, actual_channels
            # 实际顺序：args[0]=in_channels_list, args[1]=mid_ch, args[2]=naf_blocks, args[3]=upsample_factor, args[4]=reg_max, args[5]=end2end, args[6]=actual_channels
            in_channels_list = args[0] if isinstance(args[0], list) else [64]
            mid_channels = args[1] if isinstance(args[1], int) else 64
            num_naf_blocks = args[2] if isinstance(args[2], int) else 4
            upsample_factor = args[3] if isinstance(args[3], int) else 1  # 这里应该是4
            actual_channels = args[6] if args[6] is not None else in_channels_list
            
            # 如果 actual_channels 有效，优先使用它
            if actual_channels is not None and isinstance(actual_channels, list) and len(actual_channels) > 0:
                in_channels_list = actual_channels
        elif len(args) >= 4 and isinstance(args[0], list):
            # 直接调用：[in_channels_list, mid_channels, num_naf_blocks, upsample_factor]
            in_channels_list = args[0] if isinstance(args[0], list) else [args[0]]
            mid_channels = args[1] if len(args) > 1 and isinstance(args[1], int) else 64
            num_naf_blocks = args[2] if len(args) > 2 and isinstance(args[2], int) else 4
            upsample_factor = args[3] if len(args) > 3 and isinstance(args[3], int) else 4
        else:
            # 默认参数
            in_channels_list = [64] if len(args) > 0 and args[0] is not None else [64]
            mid_channels = 64
            num_naf_blocks = 4
            upsample_factor = 4  # 默认设置为4，符合您的需求

        # 确保 in_channels_list 不为 None
        if in_channels_list is None:
            in_channels_list = [64]
        
        self.num_inputs = len(in_channels_list)
        self.upsample_factor = upsample_factor
        
        # 1. 投影层：将不同通道的输入统一映射到 mid_channels
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU()
            ) for in_c in in_channels_list
        ])
        
        # 2. 特征融合层：拼接后降维
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_channels * self.num_inputs, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )
        
        # 3. 复原主体：堆叠 NAFBlocks
        self.restore_body = nn.Sequential(*[
            NAFBlock(mid_channels) for _ in range(num_naf_blocks)
        ])
        
        # 4. 输出头：映射回 3 通道，并使用 Sigmoid 限制在 [0, 1]
        self.output_head = nn.Sequential(
            nn.Conv2d(mid_channels, 3, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): 来自 Backbone 的特征图列表。
                                     假设 features[0] 是最高分辨率 (最大 H, W)。
        
        Returns:
            Tensor: 复原后的图像，形状为 [B, 3, H_target, W_target]。
                    H_target = features[0].H * upsample_factor
        """
        if not isinstance(features, list):
            features = [features]
            
        if len(features) != self.num_inputs:
            # 警告：输入数量不匹配，尝试截断或报错
            if len(features) > self.num_inputs:
                features = features[:self.num_inputs]
            else:
                raise ValueError(f"Expected {self.num_inputs} inputs, got {len(features)}")

        # 1. 投影所有特征到统一通道
        proj_feats = []
        for i, feat in enumerate(features):
            proj_feats.append(self.input_projs[i](feat))
        
        # 2. 确定目标融合分辨率 (即最高分辨率输入 features[0] 的尺寸)
        target_h, target_w = proj_feats[0].shape[2], proj_feats[0].shape[3]
        
        # 3. 上采样所有特征到目标融合分辨率
        upsampled_feats = []
        for i, pf in enumerate(proj_feats):
            if i == 0:
                upsampled_feats.append(pf)
            else:
                # 使用 bilinear 上采样，避免转置卷积
                up_pf = F.interpolate(pf, size=(target_h, target_w), mode='bilinear', align_corners=False)
                upsampled_feats.append(up_pf)
        
        # 4. 拼接并融合
        cat_feat = torch.cat(upsampled_feats, dim=1) # [B, C*num, H, W]
        fused_feat = self.fusion_conv(cat_feat)
        
        # 5. 复原处理
        restored_feat = self.restore_body(fused_feat)
        
        # 6. 生成低分辨率复原图
        output_img = self.output_head(restored_feat)
        # 7. 最终上采样到目标尺寸
        if self.upsample_factor > 1:
            output_img = F.interpolate(
                output_img, 
                scale_factor=self.upsample_factor, 
                mode='bilinear', 
                align_corners=False
            )
            
        return output_img
 
    
class RestoreBranchHasX(nn.Module):#hasX

    def __init__(self,a=0,b=0,c=0,d=0):#选择的是stage123，也就是yaml里的013
        super().__init__()

        middle_blk_num=1
        dec_blk_nums=[1,1,2]
        width = 4               # [depth, width, max_channels]
                                # n: [0.50, 0.25, 1024] 选的0.25，这里就是除以4
        self.middle_blks = nn.Sequential(*[NAFBlock(256//width) for _ in range(middle_blk_num)])

        self.upStage3To2 =  nn.Sequential(nn.Conv2d(256//width, 256 * 2//width, 1, bias=False),nn.PixelShuffle(2))
        self.decoderStage2 = nn.Sequential(*[NAFBlock(128//width) for _ in range(dec_blk_nums[2])])

        self.upStage2To1 =  nn.Sequential(nn.Conv2d(128//width, 128 * 2//width, 1, bias=False),nn.PixelShuffle(2))
        self.decoderStage1 = nn.Sequential(*[NAFBlock(64//width) for _ in range(dec_blk_nums[1])])

        self.upStage1ToImage =  nn.Sequential(nn.Conv2d(64//width, 64 * 4//width, 1, bias=False),nn.PixelShuffle(2))
        if dec_blk_nums[0] == 1:
            self.decoderStageImage = nn.Conv2d(in_channels=64//width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        else:
            pass

    def forward(self, features):

        x = self.middle_blks(features[3])

        x = self.upStage3To2(x)
        x = x + features[2]
        x = self.decoderStage2(x)

        x = self.upStage2To1(x)
        x = x + features[1]
        x = self.decoderStage1(x)

        x = self.upStage1ToImage(x)
        x = self.decoderStageImage(x)
        x = x + features[0]

        return x


class RestoreBranchNoX(nn.Module):#NoX

    def __init__(self,a=0,b=0,c=0,d=0):#选择的是stage123，也就是yaml里的013
        super().__init__()

        middle_blk_num=1
        dec_blk_nums=[1,1,2]
        width = 4               # [depth, width, max_channels]
                                # n: [0.50, 0.25, 1024] 选的0.25，这里就是除以4
        self.middle_blks = nn.Sequential(*[NAFBlock(256//width) for _ in range(middle_blk_num)])

        self.upStage3To2 =  nn.Sequential(nn.Conv2d(256//width, 256 * 2//width, 1, bias=False),nn.PixelShuffle(2))
        self.decoderStage2 = nn.Sequential(*[NAFBlock(128//width) for _ in range(dec_blk_nums[2])])

        self.upStage2To1 =  nn.Sequential(nn.Conv2d(128//width, 128 * 2//width, 1, bias=False),nn.PixelShuffle(2))
        self.decoderStage1 = nn.Sequential(*[NAFBlock(64//width) for _ in range(dec_blk_nums[1])])

        self.upStage1ToImage =  nn.Sequential(nn.Conv2d(64//width, 64 * 2//width, 1, bias=False),nn.PixelShuffle(2))
        if dec_blk_nums[0] == 1:
            self.decoderStageImage = nn.Conv2d(in_channels=32//width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        else:
            pass

    def forward(self, features):

        x = self.middle_blks(features[2])
        x = self.upStage3To2(x)
        x = x + features[1]
        x = self.decoderStage2(x)

        x = self.upStage2To1(x)
        x = x + features[0]
        x = self.decoderStage1(x)

        x = self.upStage1ToImage(x)
        x = self.decoderStageImage(x)
        #x = nn.Sigmoid()(x)
        return x