import torch
from torchvision import transforms, datasets, utils
import json
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import night_aug
from night_aug import NightAug,NightAugTransform
import night_aug
import numpy as np


def seed_torch(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch()


night_aug = NightAug(3407)
# 实例化数据增强类




# 数据预处理，定义data_transform这个字典
data_transform = {
    "train_aug": transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),  # 随机裁剪，裁剪到224*224
                                 transforms.RandomHorizontalFlip(),  # 水平方向随机翻转

                                 transforms.ToTensor(),
                                 NightAugTransform(3407),

                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    "train": transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),  # 随机裁剪，裁剪到224*224
                                 transforms.RandomHorizontalFlip(),  # 水平方向随机翻转

                                 transforms.ToTensor(),

                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    "test_aug": transforms.Compose(
                                 [transforms.Resize((224, 224), interpolation=Image.BICUBIC),  # cannot 224, must (224, 224)

                                  transforms.ToTensor(),
                                  NightAugTransform(3407),

                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),

    "test": transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
}


def train_data_aug(root_dir,batch_size):
    seed_torch()
    train_dataset = datasets.ImageFolder(root_dir+'/'+'train',data_transform['train_aug'])
    num_train_dataset = len(train_dataset)
    train_data_list = train_dataset.class_to_idx
    class_dict = dict((val,key) for key,val in train_data_list.items())

    json_string = json.dumps(class_dict,indent = 4)
    with open('class_dict.json','w') as json_file:
         json_file.write(json_string)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,
                              shuffle = True,num_workers = 0)


    return  train_loader,num_train_dataset

def train_data(root_dir,batch_size):
    seed_torch()

    train_dataset = datasets.ImageFolder(root_dir+'/'+'train',data_transform['train'])
    num_train_dataset = len(train_dataset)
    train_data_list = train_dataset.class_to_idx
    class_dict = dict((val,key) for key,val in train_data_list.items())

    json_string = json.dumps(class_dict,indent = 4)
    with open('class_dict.json','w') as json_file:
         json_file.write(json_string)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,
                              shuffle = True,num_workers = 0)


    return  train_loader,num_train_dataset


def test_data_aug(root_dir,batch_size):
    seed_torch()
    test_dataset = datasets.ImageFolder(root_dir+'/'+'test',data_transform['test_aug'])
    num_test_dataset = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,
                             shuffle = False,num_workers = 0)
    return test_loader,num_test_dataset


def test_data(root_dir,batch_size):
    seed_torch()
    test_dataset = datasets.ImageFolder(root_dir+'/'+'test',data_transform['test'])
    num_test_dataset = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,
                             shuffle = False,num_workers = 0)
    return test_loader,num_test_dataset