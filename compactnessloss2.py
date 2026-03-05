import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactnessLoss2(nn.Module):
    def __init__(self, feature_dim=64, num_features=8):

        super(CompactnessLoss2, self).__init__()
        self.feature_dim = feature_dim
        self.num_features = num_features
        # self.batch_size = batch_size
        # 初始化特征中心，需要学习的参数
        self.centers = nn.Parameter(torch.randn(num_features, feature_dim))
        # self.latent_features = nn.parameter(torch.randn((batch_size,num_features,feature_dim)))

    def forward(self, latent_features):
        device = torch.device("cuda:0")
        #进来是[128,512,1,1]
        #转成[128,8,64]

        latent_features = latent_features.view(latent_features.size(0),self.num_features,self.feature_dim).to(device)
        batch_size = latent_features.size(0)
        loss = 0.0
        for j in range(self.num_features):

            # 扩展中心以匹配批次大小
            center_expanded = self.centers[j].expand(batch_size, -1).to(device)
            # 计算潜在特征与对应中心的欧几里得距离（使用L2范数）
            loss += torch.norm(latent_features[:, j, :] - center_expanded, p=2, dim=1).pow(2)

            # cosine
            # latent_flat = latent_features[:, j, :].view(latent_features.size(0), -1)
            # center_flat = center_expanded.view(center_expanded.size(0), -1)
            #
            # # 计算余弦相似度
            # similarity = F.cosine_similarity(latent_flat, center_flat, dim=1)
            # loss += 1- similarity

        # 归一化损失值
        loss = loss / (batch_size * self.num_features)
        return loss.mean()


# # 使用示例
# # 假设我们有一个批次的潜在特征矩阵 Li，形状为 (N, M, D)
# N = 128  # 批次大小
# M = 8  # 潜在特征的数量
# D = 64  # 每个潜在特征的维度
# latent_features = torch.randn(N, M, D)  # 随机生成的潜在特征矩阵
#
# # 实例化Compactness Loss模块
# compactness_loss = CompactnessLoss(feature_dim=D, num_features=M)
#
# # 计算损失值
# loss = compactness_loss(latent_features)
# print(loss)