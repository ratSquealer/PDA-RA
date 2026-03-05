import torch.nn as nn
# from functions import ReverseLayerF
from torchvision import models
from torch.nn import functional as F
import torch

from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        resnet = models.resnet18(pretrained=False)
        checkpoint = torch.load('/data01-247/wzk1/DA/DA/DANN_orig/resnet18_msceleb.pth')
        # checkpoint = torch.load( '/data01-247/wzk1/11/AlexNet3_vit.pth')
        resnet.load_state_dict(checkpoint['state_dict'], strict=True)
        children = list(resnet.children())

        # self.pre_conv = nn.Sequential(*children[0:4])
        self.pre_conv = nn.Sequential(*children[0:4])
        self.res_block1 = children[4]
        self.res_block2 = children[5]
        self.res_block3 = children[6]
        self.res_block4 = children[7]

        self.gloabl_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.feature = nn.Sequential()
        # self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        # self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        # self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu1', nn.ReLU(True))
        # self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        # self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        # self.feature.add_module('f_drop1', nn.Dropout2d())
        # self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(512, 128))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(128, 128))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(128))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(128, 8))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(512, 128))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(128))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(128, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # self.domain__class_classifier = nn.Sequential()
        

    def forward(self, input_data, alpha):
        x_pre = self.pre_conv(input_data)

        x = self.res_block1(x_pre)
        x1 = x
        x = self.res_block2(x)
        x2 = x
        x = self.res_block3(x)
        x3 = x
        x = self.res_block4(x)
        x4 = x

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)

        x1 = F.normalize(x1.view(x1.size(0), -1), p=2, dim=1)
        x2 = F.normalize(x2.view(x2.size(0), -1), p=2, dim=1)
        x3 = F.normalize(x3.view(x3.size(0), -1), p=2, dim=1)
        x4 = F.normalize(x4.view(x4.size(0), -1), p=2, dim=1)

        x = self.gloabl_avg_pool(x)
        feature = torch.flatten(x, 1)
        x5 = x.view(x.size(0),-1)

        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        # feature = self.feature(input_data)
        # feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output,x1,x2,x3,x4,x5
