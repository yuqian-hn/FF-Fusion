import os

from PIL import Image
from torch.utils import data
from torchvision import transforms

from models.common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])


class MSRS_data(data.Dataset):
    def __init__(self, transform=to_tensor):
        super().__init__()
        self.inf_path = '/home/groupyun/桌面/sdd/Benchmark/Pb-fusion/fusion_datasets_v1/Inf'  # 获得红外路径
        self.vis_path = '/home/groupyun/桌面/sdd/Benchmark/Pb-fusion/fusion_datasets_v1/Vis'  # 获得可见光路径

        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  # 获取红外图像
        vis_image = Image.open(os.path.join(self.vis_path, name))
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        c, _, _ = vis_image.shape
        if c==3:
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        else:
            vis_y_image, vis_cb_image, vis_cr_image = vis_image, vis_image, vis_image
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name

    def __len__(self):
        return len(self.name_list)