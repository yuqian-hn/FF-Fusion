import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

class ResNeXtBackbone(nn.Module):
    """
    stem_out_stride: 4(默认) / 2 / 1
        4: 原生设置 conv1 s=2 + maxpool s=2
        2: 去掉 maxpool（只保留 conv1 s=2）
        1: conv1 改为 s=1 且去掉 maxpool
    """
    def __init__(self, pretrained=True, stem_out_stride=1):
        super().__init__()
        resnext = models.resnext101_32x8d(
            weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # 调整 stem 下采样倍率
        if stem_out_stride == 4:
            pass  # 原样：conv1 s=2, maxpool s=2
        elif stem_out_stride == 2:
            resnext.maxpool = nn.Identity()          # 只下采样 2 倍
        elif stem_out_stride == 1:
            resnext.conv1.stride = (1, 1)
            resnext.maxpool = nn.Identity()          # 不下采样
        else:
            raise ValueError("stem_out_stride must be 1, 2, or 4")

        # 拆分模块
        self.conv1 = resnext.conv1
        self.bn1   = resnext.bn1
        self.relu  = resnext.relu
        self.maxpool = resnext.maxpool  # 可能是 Identity

        self.layer1 = resnext.layer1    # 输出步长：H/4、H/2 或 H，取决于 stem_out_stride
        self.layer2 = resnext.layer2    # 后续仍是 /2 级联下采样
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

    def forward(self, x):
        _, c, _, _ = x.size()
        if c==1:
            x = torch.cat([x,x,x],dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)   # 尺寸 ≈ 输入 / {4,2,1}
        c2 = self.layer2(c1)  # 尺寸 ≈ c1 / 2
        c3 = self.layer3(c2)  # 尺寸 ≈ c2 / 2
        c4 = self.layer4(c3)  # 尺寸 ≈ c3 / 2
        return {"layer0":x,"layer1": c1, "layer2": c2, "layer3": c3, "layer4": c4}


# class ResNeXtBackbone_student(nn.Module):
#     """
#     stem_out_stride: 4(默认) / 2 / 1
#         4: 原生设置 conv1 s=2 + maxpool s=2
#         2: 去掉 maxpool（只保留 conv1 s=2）
#         1: conv1 改为 s=1 且去掉 maxpool
#     """
#     def __init__(self, pretrained=True, stem_out_stride=1):
#         super().__init__()
#         resnext = models.resnext101_32x8d(
#             weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None
#         )
#
#         # 调整 stem 下采样倍率
#         if stem_out_stride == 4:
#             pass  # 原样：conv1 s=2, maxpool s=2
#         elif stem_out_stride == 2:
#             resnext.maxpool = nn.Identity()          # 只下采样 2 倍
#         elif stem_out_stride == 1:
#             resnext.conv1.stride = (1, 1)
#             resnext.maxpool = nn.Identity()          # 不下采样
#         else:
#             raise ValueError("stem_out_stride must be 1, 2, or 4")
#
#         # 拆分模块
#         self.conv1 = resnext.conv1
#         self.bn1   = resnext.bn1
#         self.relu  = resnext.relu
#         self.maxpool = resnext.maxpool  # 可能是 Identity
#
#         self.layer1 = resnext.layer1    # 输出步长：H/4、H/2 或 H，取决于 stem_out_stride
#         self.layer2 = resnext.layer2    # 后续仍是 /2 级联下采样
#         self.layer3 = resnext.layer3
#         self.layer4 = resnext.layer4
#
#     def forward(self, vi, ir):
#         x = torch.cat([vi, ir, ir],dim=1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         c1 = self.layer1(x)   # 尺寸 ≈ 输入 / {4,2,1}
#         c2 = self.layer2(c1)  # 尺寸 ≈ c1 / 2
#         c3 = self.layer3(c2)  # 尺寸 ≈ c2 / 2
#         c4 = self.layer4(c3)  # 尺寸 ≈ c3 / 2
#         return {"layer0":x,"layer1": c1, "layer2": c2, "layer3": c3, "layer4": c4}

# 基础块：Depthwise Separable Conv (EfficientNet风格)
# class DSConv(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1):
#         super().__init__()
#         self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
#         self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
#         self.bn = nn.BatchNorm2d(out_ch)
#         self.act = nn.SiLU(inplace=True)  # 高效激活函数
#
#     def forward(self, x):
#         x = self.dw(x)
#         x = self.pw(x)
#         x = self.bn(x)
#         return self.act(x)

# EfficientNet-B0-like backbone（通道数和下采样跟 ResNet 风格对齐）
from .DBB import DiverseBranchBlock
class EffNetB0_Backbone(nn.Module):
    def __init__(self, deploy=False):
        super().__init__()
        self.deploy = deploy
        # layer0: 保持分辨率
        self.layer0 = nn.Sequential(
            DiverseBranchBlock(2, 8),
        )
        # layer1: 保持分辨率
        self.layer1 = nn.Sequential(
            DiverseBranchBlock(8, 16),
        )
        # layer2: 下采样 /2
        self.layer2 = nn.Sequential(
            # DSConv(256, 512, stride=2),
            DiverseBranchBlock(16, 32,stride=2),
        )
        # layer3: 下采样 /2
        self.layer3 = nn.Sequential(
            # DSConv(512, 1024, stride=2),
            DiverseBranchBlock(32, 64, stride=2),
        )
        # # layer4: 下采样 /2
        # self.layer4 = nn.Sequential(
        #     # DSConv(1024, 2048, stride=2),
        #     DiverseBranchBlock(512, 1024, stride=2),
        # )

    def forward(self, x):
        if self.deploy:
            self.layer0[0].switch_to_deploy()
            self.layer1[0].switch_to_deploy()
            self.layer2[0].switch_to_deploy()
            self.layer3[0].switch_to_deploy()
        c0 = self.layer0(x)   # [B,  64, H,   W  ]
        c1 = self.layer1(c0)  # [B, 256, H,   W  ]
        c2 = self.layer2(c1)  # [B, 512, H/2, W/2]
        c3 = self.layer3(c2)  # [B,1024, H/4, W/4]
        # c4 = self.layer4(c3)  # [B,2048, H/8, W/8]
        return {"layer0": c0, "layer1": c1, "layer2": c2, "layer3": c3}
#----------------下面写一个学生网络，用来学习教师网络的信息--------#
# from .DBB import DiverseBranchBlock
# class student_backbone(nn.Module):
#     def __init__(self, deploy=False):
#         super().__init__()
#         self.conv0 = nn.Sequential(
#             DiverseBranchBlock(in_channels=2, out_channels=16, kernel_size=3, deploy=deploy),
#             DiverseBranchBlock(in_channels=16, out_channels=64, kernel_size=3, deploy=deploy),
#             #nn.LeakyReLU(),
#         )
#         self.conv1 = nn.Sequential(
#             DiverseBranchBlock(in_channels=64, out_channels=128, kernel_size=3, deploy=deploy),
#             DiverseBranchBlock(in_channels=128, out_channels=256, kernel_size=3, deploy=deploy),
#             #nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.LeakyReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             DiverseBranchBlock(in_channels=256, out_channels=512, kernel_size=3, deploy=deploy),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
#             #nn.LeakyReLU(),
#         )
#         self.conv3 = nn.Sequential(
#             DiverseBranchBlock(in_channels=512, out_channels=1024, kernel_size=3, deploy=deploy),
#             nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=2, dilation=2)
#             #nn.LeakyReLU(),
#         )
#         self.conv4 = nn.Sequential(
#             DiverseBranchBlock(in_channels=1024, out_channels=2048, kernel_size=3, deploy=deploy),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.LeakyReLU(),
#         )
#
#     def forward(self, img):
#         c0 = self.conv0(img)
#         c1 = self.conv1(c0)
#         c2 = self.conv2(c1)
#         c3 = self.conv3(c2)
#         c4 = self.conv4(c3)
#         return {"layer0": c0, "layer1": c1, "layer2": c2, "layer3": c3, "layer4": c4}




# ===== 使用示例 =====
if __name__ == "__main__":
    model = ResNeXtBackbone(pretrained=True)
    x = torch.randn(1, 3, 320, 240)  # 输入图像
    feats = model(x)

    for k, v in feats.items():
        print(f"{k}: {v.shape}")
