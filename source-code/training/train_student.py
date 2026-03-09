import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from dataloder import vifs_dataloder
from models.common import gradient, clamp
from models.final_model import Fusion_Model_teacher, Fusion_Model_student
#----------------
#有两个数据集，分别在路径：    /home/groupyun/桌面/sdd/Benchmark/Pb-fusion/fusion_datasets_v1
#                         /home/groupyun/桌面/sdd/Benchmark/Pb-fusion/fusion_datasets_v2

if __name__ == '__main__':
    batch_size = 1
    workers = 1
    lr = 0.0001
    epochs = 300
    save_path = 'runs'

    train_dataset = vifs_dataloder()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Fusion_Model_teacher()
    model = model.cuda()
    model.load_state_dict(torch.load('runs/teacher.pth'))
    model.eval()
    model_student = Fusion_Model_student()
    model_student = model_student.cuda()
    model_student.train()


    optimizer = optim.Adam(model_student.parameters(), lr=lr)
    for epoch in range(epochs):
        if epoch < epochs // 2:
            lr = lr
        else:
            lr = lr * (epochs - epoch) / (epochs - epochs // 2)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        train_tqdm = tqdm(train_loader, total=len(train_loader))

        for vi_image, ir_image, name in train_tqdm:
            #print(vi_image.shape)
            vi_image = vi_image.cuda()
            ir_image = ir_image.cuda()


            optimizer.zero_grad()

            _, d1, d2, d3, d4, feats = model(vi_image, ir_image)
            fused_img, d1s, d2s, d3s, d4s, feats_s = model_student(vi_image, ir_image)
            loss_fi_gard = F.l1_loss(gradient(fused_img), torch.max(gradient(vi_image), gradient(ir_image)))
            loss_fi_pix = F.l1_loss(fused_img, torch.max(vi_image, ir_image)) + F.l1_loss(fused_img, ir_image)
            loss_mse = F.mse_loss(d1,d1s)+F.mse_loss(d2,d2s)+F.mse_loss(d3,d3s)+F.mse_loss(d4,d4s)
            loss_fi = 50 * loss_fi_gard + 10 * loss_fi_pix + 1* loss_mse

            loss = loss_fi

            loss.backward()
            optimizer.step()

            train_tqdm.set_postfix(epoch=epoch,
                                   loss_fi_gard=50*loss_fi_gard.item(),
                                   loss_fi_pix=10*loss_fi_pix.item(),
                                   loss_mse=10*loss_mse.item(),
                                   loss_total=loss.item())

        torch.save(model_student.state_dict(), f'{save_path}/student.pth')
