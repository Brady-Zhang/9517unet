import numpy as np
import torch
import random
from torch.utils.data import Dataset
import os
import cv2

class dataset(Dataset):
    def __init__(self, path):

        self.path=path

        self.images=self.path+'/image'
        self.labels=self.path+'/indexLabel'


        self.images_path = os.listdir(self.images)
        self.labels_path = os.listdir(self.labels)

    def __getitem__(self, index):

        frame=cv2.imread(self.images+'/'+self.images_path[index],1).astype("float32")#作为彩色图片读取
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255.0#由于opencv读取数据通道顺序默认为BGR，所以更改数据通道为BGR，然后再降数据归一化

        label=cv2.imread(self.labels+'/'+self.labels_path[index],0)#读取label值,作为灰度图读取

        image,label=self.RandomCrop(frame,label)#图片太大，给随机切分一下


        image = torch.tensor(image).permute(2,0,1)
        label = torch.tensor(label,dtype=torch.int64)-1#由于是从1-18,需要给转换为0-17
        return image,label

    def RandomCrop(self,img, seg):
        # 随机裁剪
        height, width, c = img.shape
        # x,y代表裁剪后的图像的中心坐标，h,w表示裁剪后的图像的高，宽，由于Unet的特性，最好给弄成2的整数
        h = 256
        w = 256
        x = random.uniform(width / 4, 3 * width / 4)
        y = random.uniform(height / 4, 3 * height / 4)

        # 左上角
        crop_xmin = np.clip(x - (w / 2), a_min=0, a_max=width).astype(np.int32)
        crop_ymin = np.clip(y - (h / 2), a_min=0, a_max=height).astype(np.int32)
        # 右下角
        crop_xmax = np.clip(x + (w / 2), a_min=0, a_max=width).astype(np.int32)
        crop_ymax = np.clip(y + (h / 2), a_min=0, a_max=height).astype(np.int32)

        cropped_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
        cropped_seg = seg[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        return cropped_img, cropped_seg

    def __len__(self):
        return len(os.listdir(self.images))

