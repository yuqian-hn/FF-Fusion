import os
import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from models.common import RGB2YCrCb
resize_size = (224,224)


class vifs_dataloder(Dataset):
    def __init__(self):
        super().__init__()
        self.to_tensor = transforms.ToTensor()
        # --- 1. 路径定义 ---
        self.vi_dir = os.path.join('/home/groupyun/桌面/sdd/Benchmark/Pb-fusion/fusion_datasets_v1/Vis')
        self.ir_dir = os.path.join('/home/groupyun/桌面/sdd/Benchmark/Pb-fusion/fusion_datasets_v1/Inf')


        #获取全部的文件名：
        self.file_list =  os.listdir(self.vi_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # 获取文件名
        file_name = self.file_list[index]
        #加载图像和标签,直接对灰度图进行处理，融合阶段的彩色用Ycrcb分解解决
        vi_image = Image.open(os.path.join(self.vi_dir, file_name)).convert('RGB').resize(resize_size)
        ir_image = Image.open(os.path.join(self.ir_dir, file_name)).convert('L').resize(resize_size)


        vi_image = self.to_tensor(vi_image)
        ir_image = self.to_tensor(ir_image)
        vi_y, cr, cb = RGB2YCrCb(vi_image)
        # gt_image = self.to_tensor(gt_image)
        return vi_y, ir_image, file_name