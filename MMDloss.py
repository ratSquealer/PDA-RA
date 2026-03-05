import torch
import torch.nn as nn


# class MMDLoss(nn.Module):
#     def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, batch_size=100):
#         super(MMDLoss, self).__init__()
#         self.kernel_mul = kernel_mul
#         self.kernel_num = kernel_num
#         self.fix_sigma = fix_sigma
#         self.batch_size = batch_size
#
#     def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
#         n_samples = int(source.size()[0]) + int(target.size()[0])
#         total = torch.cat([source, target], dim=0)
#
#         total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#
#         L2_distance = ((total0 - total1) ** 2).sum(2)
#         if fix_sigma:
#             bandwidth = fix_sigma
#         else:
#             bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
#         bandwidth /= kernel_mul ** (kernel_num // 2)
#         bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
#
#         kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#         kernel_val = sum(kernel_val)
#         return kernel_val.mean()
#
#     def forward(self, source, target):
#         # Divide the source and target features into smaller batches
#         fragments = max(int(source.size(0) / self.batch_size), 1)
#         loss = 0
#         for i in range(fragments):
#             source_batch = source[i * self.batch_size:(i + 1) * self.batch_size, :]
#             target_batch = target[i * self.batch_size:(i + 1) * self.batch_size, :]
#             loss += self.guassian_kernel(source_batch, target_batch, self.kernel_mul, self.kernel_num, self.fix_sigma)
#
#         return loss / fragments


# # 使用示例
# # 初始化MMD损失函数
# mmd_loss = MMDLoss(batch_size=10)  # 假设每批处理10个样本
#
# # 假设feature_map1和feature_map2是你想要约束的两个特征图
# feature_map1 = torch.randn(100, 64)  # 100个样本，每个样本64特征
# feature_map2 = torch.randn(100, 64)
#
# # 计算MMD损失
# loss = mmd_loss(feature_map1, feature_map2)
#
# print("MMD Loss:", loss.item())
import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=1.5, kernel_num=4, fix_sigma=None, batch_size=20):
        super(MMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.batch_size = batch_size

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # Use torch broadcasting rather than unsqueeze and expand
        L2_distance = (total.unsqueeze(1) - total.unsqueeze(0)).pow(2).sum(2)

        # Avoid creating multiple copies of L2_distance
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)

        kernel_val = 0
        for i in range(kernel_num):
            bandwidth_temp = bandwidth * (kernel_mul ** i)
            kernel_val += torch.exp(-L2_distance / bandwidth_temp)

        return kernel_val.mean()
        # return kernel_val.sum()

    def forward(self, source, target):
        fragments = max(int(source.size(0) / self.batch_size), 1)#数据集划分成多少个批次
        loss = 0
        for i in range(fragments):
            source_batch = source[i * self.batch_size:(i + 1) * self.batch_size, :]#从 source 张量中提取第 i 个批次的数据样本
            target_batch = target[i * self.batch_size:(i + 1) * self.batch_size, :]
            loss += self.guassian_kernel(source_batch, target_batch, self.kernel_mul, self.kernel_num, self.fix_sigma)

        return loss / fragments


#
# class MMDLoss(nn.Module):
#     def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#         super(MMDLoss, self).__init__()
#         self.kernel_mul = kernel_mul
#         self.kernel_num = kernel_num
#         self.fix_sigma = fix_sigma
#
#
#
#     def guassian_kernel(self,source, target, kernel_mul, kernel_num, fix_sigma=None):
#         n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
#         total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
#         # 将total复制（n+m）份
#         total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
#         total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
#         L2_distance = ((total0 - total1) ** 2).sum(2)
#         # 调整高斯核函数的sigma值
#         if fix_sigma:
#             bandwidth = fix_sigma
#         else:
#             bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
#         # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
#         bandwidth /= kernel_mul ** (kernel_num // 2)
#         bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
#         # 高斯核函数的数学表达式
#         kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#         # 得到最终的核矩阵
#         return sum(kernel_val)  # /len(kernel_val)
#
#
#     def forward(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#
#         batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
#         kernels = self.guassian_kernel(source, target,
#                                   kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#         # 根据式（3）将核矩阵分成4部分
#         XX = kernels[:batch_size, :batch_size]
#         YY = kernels[batch_size:, batch_size:]
#         XY = kernels[:batch_size, batch_size:]
#         YX = kernels[batch_size:, :batch_size]
#         loss = torch.mean(XX + YY - XY - YX)
#         return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算

# class MMDLoss(nn.Module):
#     def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#         super(MMDLoss, self).__init__()
#         self.kernel_mul = kernel_mul
#         self.kernel_num = kernel_num
#         self.fix_sigma = fix_sigma
#
#     def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma=None):
#         n_samples = int(source.size()[0]) + int(target.size()[0])
#         total = torch.cat([source, target], dim=0)
#         total0 = total.unsqueeze(0)
#         total1 = total.unsqueeze(1)
#         L2_distance = (total0 - total1).pow_(2).sum(2)
#         if fix_sigma:
#             bandwidth = fix_sigma
#         else:
#             bandwidth = L2_distance.mean()
#         bandwidth /= kernel_mul ** (kernel_num // 2)
#         bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
#         kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#         return sum(kernel_val)
#
#     def forward(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#         batch_size = int(source.size()[0])
#         kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
#         XX = kernels[:batch_size, :batch_size]
#         YY = kernels[batch_size:, batch_size:]
#         XY = kernels[:batch_size, batch_size:]
#         YX = kernels[batch_size:, :batch_size]
#         loss = XX.add(YY).sub_(XY).sub_(YX).mean()
#         return loss

# import torch
# import torch.nn as nn
#
#
# class MMDLoss(nn.Module):
#     def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#         super(MMDLoss, self).__init__()
#         self.kernel_mul = kernel_mul
#         self.kernel_num = kernel_num
#         self.fix_sigma = fix_sigma
#
#     def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma=None, batch_size=25):
#         n_samples = int(source.size()[0]) + int(target.size()[0])
#         total = torch.cat([source, target], dim=0)
#
#         # Initialize the kernel matrix with zeros
#         kernel_matrix = torch.zeros((n_samples, n_samples), dtype=total.dtype, device=total.device)
#
#         for i in range(0, n_samples, batch_size):
#             s1 = slice(i, min(n_samples, i + batch_size))
#             total0_batch = total[s1].unsqueeze(0)
#             for j in range(0, n_samples, batch_size):
#                 s2 = slice(j, min(n_samples, j + batch_size))
#                 total1_batch = total[s2].unsqueeze(1)
#
#                 L2_distance_batch = ((total0_batch - total1_batch) ** 2).sum(2)
#                 if fix_sigma:
#                     bandwidth = fix_sigma
#                 else:
#                     bandwidth = torch.sum(L2_distance_batch.data) / (batch_size * n_samples)
#                 bandwidth /= kernel_mul ** (kernel_num // 2)
#                 bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
#
#                 # Compute the kernel values for the batch and add them to the kernel matrix
#                 kernel_val_batch = [torch.exp(-L2_distance_batch / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#                 kernel_matrix[s1, s2] = sum(kernel_val_batch)
#
#         return kernel_matrix
#
#     def forward(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, batch_size=25):
#         batch_size = int(source.size()[0])
#         kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
#                                        fix_sigma=fix_sigma, batch_size=batch_size)
#         XX = kernels[:batch_size, :batch_size]
#         YY = kernels[batch_size:, batch_size:]
#         XY = kernels[:batch_size, batch_size:]
#         YX = kernels[batch_size:, :batch_size]
#         loss = torch.mean(XX + YY - XY - YX)
#         return loss

# Example usage:
# Initialize the MMDLoss object
# mmd_loss = MMDLoss()

# Your source and target tensors here (dummy tensors for example purposes)
# source = torch.randn(100, 64)  # 100 samples with 64 features each
# target = torch.randn(100, 64)  # 100 samples with 64 features each

# Compute the loss
# loss_value = mmd_loss(source, target)
# print(loss_value)
