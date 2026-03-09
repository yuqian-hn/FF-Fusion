"""测试融合网络"""
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloder_test import MSRS_data
from models.common import clamp, YCrCb2RGB
from models_piafusion.fusion_model import PIAFusion
import os
import time




if __name__ == '__main__':

    num_works = 1
    fusion_result_path = 'res1'

    if not os.path.exists(fusion_result_path):
        os.makedirs(fusion_result_path)


    test_dataset = MSRS_data()
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(fusion_result_path):
        os.makedirs(fusion_result_path)

    #######加载模型
    model =(PIAFusion())

    model.load_state_dict(torch.load('runs_piafusion/fusion_model_epoch_29.pth', map_location='cpu'))

    model.eval()


    ##########加载数据
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name in test_tqdm:
            vis_y_image = vis_y_image
            cb = vis_cb_image
            cr = vis_cr_image
            inf_image = inf_image

            #####模型推理
            #print(vis_y_image.shape, inf_image.shape)
            start = time.time()
            f= model(vis_y_image, inf_image)
            end = time.time()
            print('time:', end-start)


            ###########转为rgb
            fused = clamp(f)
            if torch.equal(vis_y_image, cb):
                rgb_fused_image = fused[0]
            else:
                rgb_fused_image = YCrCb2RGB(fused[0], cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
            rgb_fused_image.save(f'{fusion_result_path}/{name[0]}')
