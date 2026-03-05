import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    def forward(self, features, labels):
        device = torch.device("cuda:0")
        batch_size = features.size(0)
        self.centers = nn.Parameter(self.centers.to(device))
        # 计算特征与中心点的差值
        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
        #           torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # # distmat.addmm_(1, -2, features, self.centers.t())
        # distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        # features的形状是[128, 512, 1, 1]，首先要将它减少到[128, 512]
        features = features.view(features.size(0), -1).to(device)
        # 计算特征的平方和，结果形状将是[128, 1]
        features_sum_sq = torch.pow(features, 2).sum(dim=1, keepdim=True).to(device)
        # 现在可以扩展features_sum_sq，因为它的形状是[128, 1]
        features_sum_sq = features_sum_sq.expand(batch_size, self.num_classes)  # [128, num_classes]

        # self.centers的形状应该是[num_classes, 512]
        # 计算centers的平方和，结果形状将是[num_classes, 1]
        centers_sum_sq = torch.pow(self.centers, 2).sum(dim=1, keepdim=True)
        # 扩展并转置，以便它的形状是[128, num_classes]
        centers_sum_sq = centers_sum_sq.t().expand(batch_size, self.num_classes)

        # 初始化distmat为features_sum_sq + centers_sum_sq
        distmat = features_sum_sq + centers_sum_sq

        # features的形状现在是[128, 512]
        # 执行矩阵乘法和加法操作
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)#平方差公式



        classes = torch.arange(self.num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes).to(device)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss,self.centers


# # 假设 num_classes 是类别总数，feat_dim 是特征维度
# num_classes = 10
# feat_dim = 2
# center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
# cross_entropy_loss = nn.CrossEntropyLoss()
#
# # features 是网络提取的特征，labels 是真实标签
# # features 的维度是 (batch_size, feat_dim)，labels 的维度是 (batch_size)
# features = torch.randn(32, feat_dim)
# labels = torch.randint(high=num_classes, size=(32,))
#
# # 计算损失
# loss_center = center_loss(features, labels)
# loss_cross_entropy = cross_entropy_loss(features, labels)
#
# # 结合两种损失
# lambda_center = 0.5
# loss = loss_cross_entropy + lambda_center * loss_center
#
# # 反向传播
# loss.backward()