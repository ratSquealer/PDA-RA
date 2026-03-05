import torch
import torch.nn as nn
import torch.nn.functional as F


#直接用均值，不更新，效果不好

class CompactnessLoss1(nn.Module):
    def __init__(self):
        super(CompactnessLoss1, self).__init__()

    def forward(self, features, labels):
        #features: 特征矩阵，大小为 (batch_size, feature_dim)
        #labels: 标签，大小为 (batch_size,)
        device = torch.device("cuda:1")
        features = features.view(features.size(0), -1).to(device)
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        loss = 0.0

        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]

            # 计算类中心
            class_center = torch.mean(class_features, dim=0, keepdim=True)

            # 计算类内距离的平方和
            dists = torch.sum((class_features - class_center) ** 2)

            # 累加到loss中
            loss += dists

        # 对类别数量进行归一化
        loss /= num_classes
        return loss