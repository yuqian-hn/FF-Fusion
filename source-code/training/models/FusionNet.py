import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """标准卷积块 Conv-BN-ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ae_fusion_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 对不同layer做通道压缩，使得解码时可以对齐
        self.reduce0 = ConvBlock(64, 64)
        self.reduce1 = ConvBlock(256, 128)
        self.reduce2 = ConvBlock(512, 256)
        self.reduce3 = ConvBlock(1024, 512)
        self.reduce4 = ConvBlock(2048, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   # from layer4
        self.dec4 = ConvBlock(512 + 512, 512)  # concat with layer3

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # from dec4
        self.dec3 = ConvBlock(256 + 256, 256)  # concat with layer2

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # from dec3
        self.dec2 = ConvBlock(128 + 128, 128)  # concat with layer1

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # from dec2
        self.dec1 = ConvBlock(64 + 64, 64)    # concat with layer0

        # Output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, feats):
        c0 = self.reduce0(feats["layer0"])   # [B,64,320,240]
        c1 = self.reduce1(feats["layer1"])   # [B,128,320,240]
        c2 = self.reduce2(feats["layer2"])   # [B,256,160,120]
        c3 = self.reduce3(feats["layer3"])   # [B,512,80,60]
        c4 = self.reduce4(feats["layer4"])   # [B,512,40,30]

        # Decoder
        u4 = self.up4(c4)                    # [B,512,80,60]
        d4 = self.dec4(torch.cat([u4, c3], dim=1))

        u3 = self.up3(d4)                    # [B,256,160,120]
        d3 = self.dec3(torch.cat([u3, c2], dim=1))

        u2 = self.up2(d3)                    # [B,128,320,240]
        d2 = self.dec2(torch.cat([u2, c1], dim=1))

        u1 = self.up1(d2)                    # [B,64,640,480]  ← 注意这里会比输入大
        # 为了对齐回原分辨率(320x240)，需要调整stride或插值
        u1 = nn.functional.interpolate(u1, size=c0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, c0], dim=1))

        out = self.out_conv(d1)              # [B,1,320,240]

        return out, d1, d2, d3, d4, feats

#----------写一个用于学生网络的fusion_head-----#
from .DBB import DiverseBranchBlock
class ae_fusion_Net_student(nn.Module):
    def __init__(self, deploy=False):
        super().__init__()
        # 对不同layer做通道压缩，使得解码时可以对齐
        self.reduce0 = DiverseBranchBlock(64, 64, deploy=deploy)
        self.reduce1 = DiverseBranchBlock(256, 128, deploy=deploy)
        self.reduce2 = DiverseBranchBlock(512, 256, deploy=deploy)
        self.reduce3 = DiverseBranchBlock(1024, 512, deploy=deploy)
        self.reduce4 = DiverseBranchBlock(2048, 512, deploy=deploy)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   # from layer4
        self.dec4 = DiverseBranchBlock(512 + 512, 512, deploy=deploy)  # concat with layer3

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # from dec4
        self.dec3 = DiverseBranchBlock(256 + 256, 256, deploy=deploy)  # concat with layer2

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # from dec3
        self.dec2 = DiverseBranchBlock(128 + 128, 128, deploy=deploy)  # concat with layer1

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # from dec2
        self.dec1 = DiverseBranchBlock(64 + 64, 64, deploy=deploy)    # concat with layer0

        # Output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, feats):
        self.reduce0.switch_to_deploy()
        self.reduce1.switch_to_deploy()
        self.reduce2.switch_to_deploy()
        self.reduce3.switch_to_deploy()
        self.reduce4.switch_to_deploy()
        self.dec4.switch_to_deploy()
        self.dec3.switch_to_deploy()
        self.dec2.switch_to_deploy()
        self.dec1.switch_to_deploy()

        c0 = self.reduce0(feats["layer0"])   # [B,64,320,240]
        c1 = self.reduce1(feats["layer1"])   # [B,128,320,240]
        c2 = self.reduce2(feats["layer2"])   # [B,256,160,120]
        c3 = self.reduce3(feats["layer3"])   # [B,512,80,60]
        c4 = self.reduce4(feats["layer4"])   # [B,512,40,30]

        # Decoder
        u4 = self.up4(c4)                    # [B,512,80,60]
        d4 = self.dec4(torch.cat([u4, c3], dim=1))

        u3 = self.up3(d4)                    # [B,256,160,120]
        d3 = self.dec3(torch.cat([u3, c2], dim=1))

        u2 = self.up2(d3)                    # [B,128,320,240]
        d2 = self.dec2(torch.cat([u2, c1], dim=1))

        u1 = self.up1(d2)                    # [B,64,640,480]  ← 注意这里会比输入大
        # 为了对齐回原分辨率(320x240)，需要调整stride或插值
        u1 = nn.functional.interpolate(u1, size=c0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, c0], dim=1))

        out = self.out_conv(d1)              # [B,1,320,240]

        return out, d1, d2, d3, d4, feats

