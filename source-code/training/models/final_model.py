import torch
import torch.nn
import torch.nn.functional as F
from torch import nn
from .backbone import ResNeXtBackbone
from .cross_att_fusion import Cross_attention
from .wtconv2d import WTConv2d_VIF
from .FusionNet import ae_fusion_Net, ae_fusion_Net_student

class Fusion_Model_teacher(nn.Module):
    def __init__(self):
        super(Fusion_Model_teacher, self).__init__()
        self.backbone = ResNeXtBackbone(pretrained=True)
        self.shallow1 = WTConv2d_VIF(in_channels=64, out_channels=64)
        self.shallow2 = WTConv2d_VIF(in_channels=256, out_channels=256)
        self.seg1 = Cross_attention(in_channel=512)
        self.seg2 = Cross_attention(in_channel=1024)
        self.seg3 = Cross_attention(in_channel=2048)

        self.fusion_task_head = ae_fusion_Net()


    def forward(self, vi, ir):
        feat_vi = self.backbone(vi)
        feat_ir = self.backbone(ir)
        '''
        feat特征有5层，分别是layer 0,1,2,3,4
        layer0: torch.Size([1, 64, 320, 240])
        layer1: torch.Size([1, 256, 320, 240])
        layer2: torch.Size([1, 512, 160, 120])
        layer3: torch.Size([1, 1024, 80, 60])
        layer4: torch.Size([1, 2048, 40, 30])
        '''
        feat_f = feat_vi
        feat_f['layer0'] = self.shallow1(feat_vi['layer0'], feat_ir['layer0'])
        feat_f['layer1'] = self.shallow2(feat_vi['layer1'], feat_ir['layer1'])
        feat_f['layer2'] = self.seg1(feat_vi['layer2'], feat_ir['layer2'])
        feat_f['layer3'] = self.seg2(feat_vi['layer3'], feat_ir['layer3'])
        feat_f['layer4'] = self.seg3(feat_vi['layer4'], feat_ir['layer4'])

        fused_img, d1, d2, d3, d4, feats = self.fusion_task_head(feat_f)
        return fused_img, d1, d2, d3, d4, feats


#-----------写一个学生网络-----------------#
from .backbone import EffNetB0_Backbone
class Fusion_Model_student(nn.Module):
    def __init__(self, deploy=False):
        super(Fusion_Model_student, self).__init__()
        self.backbone = EffNetB0_Backbone()
        self.fusion_task_head = ae_fusion_Net_student(deploy=deploy)

    def forward(self, vi, ir):
        feat = self.backbone(torch.cat((vi, ir), dim=1))
        fused_img, d1, d2, d3, d4, feats = self.fusion_task_head(feat)

        return fused_img, d1, d2, d3, d4, feats






