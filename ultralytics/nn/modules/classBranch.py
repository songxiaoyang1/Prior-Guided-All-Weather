import torch
from torch import nn


class ClassBranchNoX(nn.Module):
    def __init__(self, a=0, b=0, c=0, d=0):
        super().__init__()
        in_channels = 1024 // 4  #    # [depth, width, max_channels]
        # n: [0.50, 0.25, 1024] 选的0.25
        hidden_dim = 256 // 4
        nc_weather = 4
        dropout_rate = 0.2
        # 全局平均池化，消除空间维度 [B,C,H,W] -> [B,C]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 分类MLP头
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, nc_weather)

    def forward(self, backbone_feat10):
        """backbone_feat10: 主干第10层C2PSA输出 [B,1024,H/32,W/32] return: weather_logits [B, nc_weather] 未softmax.
        """
        # 全局池化
        x = self.gap(backbone_feat10[0])  # [B,1024,1,1]
        x = torch.flatten(x, 1)  # [B,1024]

        # MLP前向
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        output_cls = self.fc2(x)

        return output_cls


class ClassBranchHasX(nn.Module):
    def __init__(self, a=0, b=0, c=0, d=0):
        super().__init__()
        in_channels = 1024 // 4  #    # [depth, width, max_channels]
        # n: [0.50, 0.25, 1024] 选的0.25
        hidden_dim = 256 // 4
        nc_weather = 4
        dropout_rate = 0.2
        # 全局平均池化，消除空间维度 [B,C,H,W] -> [B,C]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 分类MLP头
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, nc_weather)

    def forward(self, backbone_feat11):
        """backbone_feat11: 主干第11层C2PSA输出 [B,1024,H/32,W/32] return: weather_logits [B, nc_weather] 未softmax.
        """
        # 全局池化
        x = self.gap(backbone_feat11[0])  # [B,1024,1,1]
        x = torch.flatten(x, 1)  # [B,1024]

        # MLP前向
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        output_cls = self.fc2(x)

        return output_cls
