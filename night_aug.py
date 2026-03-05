import torch
import torchvision.transforms as T
import numpy as np
from numpy import random as R
# import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import random




class NightAug:
    def __init__(self,seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.gaussian = T.GaussianBlur(11, (0.1, 2.0))
        # self.set_seed(seed)

    # def mask_img(self, img, cln_img):
    #     while R.random() > 0.4:
    #         x1 = R.randint(img.shape[1])
    #         x2 = R.randint(img.shape[1])
    #         y1 = R.randint(img.shape[2])
    #         y2 = R.randint(img.shape[2])
    #         img[:, x1:x2, y1:y2] = cln_img[:, x1:x2, y1:y2]
    #     return img
    # def set_seed(self,seed_value):
    #     random.seed(seed_value)  # Python的内置random模块
    #     R.seed(seed_value)  # Numpy的随机模块
    #     torch.manual_seed(seed_value)  # PyTorch的CPU运算
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed_value)


    # def gaussian_heatmap(self, x):
    #     """
    #     It produces single gaussian at a random point
    #     """
    #     sig = torch.randint(low=1, high=10, size=(1,))[0]
    #
    #     image_size = x.shape[1:]
    #     center = (torch.randint(image_size[0], (1,))[0], torch.randint(image_size[1], (1,))[0])
    #     x_axis = torch.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
    #     y_axis = torch.linspace(0, image_size[1] - 1, image_size[1]) - center[1]
    #     xx, yy = torch.meshgrid(x_axis, y_axis)
    #     kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
    #     new_img = (x * (1 - kernel) + 255 * kernel).type(torch.uint8)
    #     return new_img

    def gaussian_heatmap(self, x):
        """
        It produces a single gaussian at a random point
        """
        sig = torch.randint(low=1, high=10, size=(1,))[0]
        image_size = x.shape[1:]
        center = (torch.randint(image_size[0], (1,))[0], torch.randint(image_size[1], (1,))[0])
        x_axis = torch.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
        y_axis = torch.linspace(0, image_size[1] - 1, image_size[1]) - center[1]

        # 添加indexing='ij'参数
        xx, yy = torch.meshgrid(x_axis, y_axis)#, indexing='ij'

        kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
        new_img = (x * (1 - kernel) + 3 * kernel)
        return new_img
    import torch

    # def gaussian_heatmap(self, x):
    #     """
    #     It produces a single gaussian at a random point
    #     """
    #     # 使用更合适的sig范围，例如图像尺寸的一小部分
    #
    #     image_size = x.shape[1:]
    #     sig = torch.randint(low=1, high=image_size[0] // 8, size=(1,))[0]
    #     center = (
    #     torch.randint(low=0, high=image_size[0], size=(1,))[0], torch.randint(low=0, high=image_size[1], size=(1,))[0])
    #     x_axis = torch.linspace(0, image_size[0] - 1, steps=image_size[0]) - center[0]
    #     y_axis = torch.linspace(0, image_size[1] - 1, steps=image_size[1]) - center[1]
    #
    #     xx, yy = torch.meshgrid(x_axis, y_axis, indexing='ij')
    #
    #     kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
    #
    #     # 归一化高斯核，使得其总和为1
    #     kernel /= kernel.sum()
    #
    #     # 应用高斯核，重点在于增加的亮度应该是高斯核的加权，而非直接相加
    #     new_img = x + (200 - x) * kernel.unsqueeze(0)
    #     # 首先，对kernel进行归一化处理
    #
    #     # 保证图像的像素值在合理的范围内
    #     new_img = torch.clamp(new_img, min=50, max=200)
    #
    #     return new_img
    import torch

    # def gaussian_heatmap(self, x):
    #     """
    #     It produces a gaussian heatmap for a multi-channel image.
    #     """
    #     sig = torch.randint(low=1, high=50, size=(1,)).item()  # 根据实际情况调整
    #     num_channels, height, width = x.shape  # 假设x的形状是[channels, height, width]
    #     center = (torch.randint(low=0, high=height, size=(1,)).item(),
    #               torch.randint(low=0, high=width, size=(1,)).item())
    #
    #     x_axis = torch.linspace(-center[0], height - 1 - center[0], steps=height)
    #     y_axis = torch.linspace(-center[1], width - 1 - center[1], steps=width)
    #
    #     xx, yy = torch.meshgrid(x_axis, y_axis, indexing='ij')
    #
    #     sig = torch.tensor(sig, dtype=torch.float32)  # 将 sig 转换为 Tensor
    #     kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
    #     kernel /= kernel.sum()  # 归一化高斯核
    #
    #     # 扩展高斯核到所有通道
    #     kernel = kernel.expand(num_channels, -1, -1)
    #
    #     # 应用高斯核到多通道图像
    #     new_img = x + (255 - x) * kernel
    #
    #     # 限制像素值在0-255之间
    #     new_img = torch.clamp(new_img, 0, 255).type(torch.uint8)
    #
    #     return new_img

    def aug(self, sample):


        img = sample['image']
        g_b_flag = True
        # R.seed(seed_value)

        # Guassian Blur
        if R.random() > 0.5:
            img = self.gaussian(img)

        cln_img_zero = img.detach().clone()

        # Gamma
        if R.random() > 0.5:
            cln_img = img.detach().clone()
            val = 1 / (R.random() * 0.2 + 0.2)
            img = T.functional.adjust_gamma(img, val)
            # img = self.mask_img(img, cln_img)
            g_b_flag = False

        # Brightness
        if R.random() > 0.5 or g_b_flag:
            cln_img = img.detach().clone()
            val = R.random() * 0.2 + 0.5
            img = T.functional.adjust_brightness(img, val)
            # img = self.mask_img(img, cln_img)

        # Contrast
        if R.random() > 0.5:
            cln_img = img.detach().clone()
            val = R.random() * 0.3 + 0.2
            img = T.functional.adjust_contrast(img, val)
            # img = self.mask_img(img, cln_img)
        # img = self.mask_img(img, cln_img_zero)


        # prob = 0.5
        # while R.random() > prob:
        # if R.random() > 0.5:
        #     img = self.gaussian_heatmap(img)
        #     img = self.gaussian_heatmap(img)
        #     prob += 0.1

      #  Noise
      #   if R.random() > 0.5:
      #       std_dev = R.uniform(0.2, 0.3)
      #       n = torch.clamp(torch.normal(0.1, std_dev, img.shape), min=0)
      #       img = n + img
      #       img = torch.clamp(img, max=255).type(torch.uint8)

        # if R.random() > 0.5:
        #     # 生成噪声后立即应用clamp确保没有负值
        #     n = torch.normal(5, R.randint(10, 15), img.shape).clamp(min=0).to(img.dtype)
        #     img = (n + img).clamp(max=255).type(torch.uint8)


        # Check whether to add noise

        # 判断是否添加噪声
        if R.random() > 0.5:
            # 定义噪声水平：在一个非常小的范围内的均匀分布
            noise_level = R.uniform(0.01, 0.03)  # 这样可以保持噪声非常低

            # 生成在范围 [-noise_level, noise_level] 内的均匀噪声
            noise = torch.rand(img.shape) * (2 * noise_level) - noise_level

            # 在缩放到 [0, 255] 范围后确保噪声在 [0, 1] 范围内，然后再缩放
            img = img.float()+noise
            img = torch.clamp(img, 0, 255)

            # 将图像数据类型转换回适合图像的 uint8 类型
            # img = img.type(torch.uint8)




        sample['image'] = img
        return sample

# class NightAugTransform:
#     def __init__(self):
#         self.night_aug = NightAug()
#
#     def __call__(self, img):
#         if not img.dtype.is_floating_point:
#             img = img.float()
#         sample = {'image': img}
#         augmented_sample = self.night_aug.aug(sample)
#         return augmented_sample['image']

class NightAugTransform:
    def __init__(self,seed):
        self.night_aug = NightAug(seed = seed)

    def __call__(self, img):
        # 确保输入img是浮点类型张量
        if not img.dtype.is_floating_point:
            img = img.float()

        # 创建一个包含图像的字典结构，以满足NightAug的API期望
        sample = {'image': img}

        # 应用夜间增强效果
        augmented_sample = self.night_aug.aug(sample)

        # 取出增强后的图像，确保它依然是浮点类型
        augmented_image = augmented_sample['image']
        if not augmented_image.dtype.is_floating_point:
            augmented_image = augmented_image.float()

        return augmented_image


# 自定义数据集
# class CustomDataset(Dataset):
#     def __init__(self, data, transform=None):
#         self.data = data
#         self.transform = transform
#
#     def __getitem__(self, index):
#         # 获取数据样本
#         x = self.data[index]
#
#         # 如果有转换，则应用它
#         if self.transform:
#             x = self.transform(x)
#
#         return x
#
#     def __len__(self):
#         return len(self.data)


