import torch
import torch.nn as nn
from functools import partial
import yaml

import argparse
import os
import random
import shutil
from os.path import join

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm

from torch import randn
from torch.nn import init

from LabelSmoothing import LabelSmoothingLoss

import math
from conv2Fpn import *
from sklearn.metrics import recall_score, f1_score
from loss.contrastive import BalSCL
from loss.logitadjust import LogitAdjust
from randaugment import rand_augment_transform
from PIL import Image
import torch



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 取 topk 个预测值
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
 

#######################
##### 1 - Setting #####
#######################

##### args setting
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='', help='dataset dir')
parser.add_argument('-b', '--batch_size', default=16, help='batch_size')
parser.add_argument(
    '-g', '--gpu', default='0', help='example: 0 or 1, to use different gpu'
)
parser.add_argument('-w', '--num_workers', default=12, help='num_workers of dataloader')
parser.add_argument('-s', '--seed', default=42, help='random seed')
parser.add_argument(
    '-n',
    '--note',
    default='',
    help='exp note, append after exp folder, fgvc(_r50) for example',
)
parser.add_argument(
    '-a',
    '--amp',
    default=0,
    help='0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp',
)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--beta', default=0.42, type=float, help='supervised contrastive loss weight')##0.35
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
args = parser.parse_args()



##### exp setting
seed = int(args.seed)
datasets_dir = args.dir
nb_epoch = 128  # 128 as default to suit scheduler
batch_size = int(args.batch_size)
num_workers = int(args.num_workers)
# lr_begin = (batch_size / 256) * 0.1  # learning rate at begining
lr_begin = 0.00172
use_amp = int(args.amp)  # use amp to accelerate training

##### data settings
current_dir = os.getcwd()  # 获取当前目录的路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # 获取上级目录的绝对路径
# data_dir = join(parent_dir, 'data')
data_dir = './data_from_web/Herb-LT_split42'
# data_dir = '/nfs/home/liuxinyao/data/CUB_200_2011'

# data_dir = join('data_mixed', datasets_dir)
data_sets = ['train', 'test']
num_classes = len(
    os.listdir(join(data_dir, data_sets[0]))
)  # get number of class via img folders automatically
exp_dir = 'result_OURS/{}{}'.format(datasets_dir, args.note)  # the folder to save model

##### Dataloader setting
re_size = 400
crop_size = 384
train_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.RandomCrop(crop_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
# 使用 ImageFolder 加载数据集
train_set = ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
test_set = ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)

# 使用 DataLoader 加载数据
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
rgb_mean = (0.485, 0.456, 0.406)
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
]

augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
]

transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                                      transforms.Compose(augmentation_sim), ]

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name='Metric', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
            # Randomly select a class and retrieve a sample from that class
            label = random.randint(0, self.num_classes - 1)
            index = random.choice(self.class_data[label])
            path = self.img_path[index]
        else:
            path = self.img_path[index]
            label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](sample)
                sample2 = self.transform[1](sample)
                sample3 = self.transform[2](sample)
                return [sample1, sample2, sample3], label  # , index
            else:
                return self.transform(sample), label




##### CUDA device setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


##### Random seed setting
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

########################### 模 型 #######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = CL_Prompt_Fpn(pretrained=True)
net = net.to(device)  # 放到 GPU

num_classes = len(train_set.classes)  # 一定要和实际数据集一致

# =============================
# 1. 获取特征维度
# =============================
dummy_input = torch.randn(1, 3, 224, 224).cuda()  # 用你的输入尺寸
with torch.no_grad():
    # net.forward_features 需返回 fc 前的特征，如果没有，可用 net(inputs) 返回第二个输出 feat_mlp
    feat = net.forward_features(dummy_input)  # 如果 CL_Prompt_Fpn 没有 forward_features，可改用 net(dummy_input)[1]
    in_features = feat.shape[1]

# =============================
# 2. 替换 fc 层
# =============================
net.fc = NormedLinear(in_features, num_classes).cuda()

# =============================
# 3. 替换 head
# =============================
if hasattr(net, 'head') and isinstance(net.head, nn.Sequential):
    # 找最后一个 Linear
    last_linear_idx = None
    for idx, m in reversed(list(enumerate(net.head))):
        if isinstance(m, nn.Linear):
            last_linear_idx = idx
            break
    if last_linear_idx is not None:
        in_features = net.head[last_linear_idx].in_features
        net.head[last_linear_idx] = nn.Linear(in_features, num_classes).cuda()

# =============================
# 4. 替换 head_fc
# =============================
if hasattr(net, 'head_fc') and isinstance(net.head_fc, nn.Sequential):
    last_linear_idx = None
    for idx, m in reversed(list(enumerate(net.head_fc))):
        if isinstance(m, nn.Linear):
            last_linear_idx = idx
            break
    if last_linear_idx is not None:
        in_features = net.head_fc[last_linear_idx].in_features
        net.head_fc[last_linear_idx] = nn.Linear(in_features, num_classes).cuda() 
###########################################################


# for param in net.parameters():
#     param.requir es_grad = True  # make parameters in model learnable


##### optimizer setting

# 获取每个类别样本数
cls_num_list = [0] * len(train_set.classes)
for _, label in train_set.samples:
    cls_num_list[label] += 1

ABCL_loss = BalSCL(cls_num_list=cls_num_list, temperature=args.temp).cuda()

LSLoss = LabelSmoothingLoss(
    classes=num_classes, smoothing=0.1
)  # label smoothing to improve performance
optimizer = torch.optim.SGD( 
    net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4  ##5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128)##128


##### file/folder prepare
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

shutil.copyfile('train_conv.py', exp_dir + '/train_conv.py')
shutil.copyfile('LabelSmoothing.py', exp_dir + '/LabelSmoothing.py')
shutil.copyfile('conv2Fpn.py', exp_dir + '/conv2Fpn.py')

with open(os.path.join(exp_dir, 'train_log.csv'), 'w+') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc, Test_Acc\n')
    file.flush()



if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    ########################
    ##### 2 - Training #####
    ########################
    net.cuda()
    min_train_loss = float('inf')
    max_eval_acc = 0

    for epoch in range(nb_epoch):
        top1 = AverageMeter('Acc@1', ':6.2f')
        print('\n===== Epoch: {} ====='.format(epoch))
        net.train()  # set model to train mode, enable Batch Normalization and Dropout
        lr_now = optimizer.param_groups[0]['lr']
        train_loss = train_correct = train_total = idx = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, ncols=80)):
            idx = batch_idx

            if inputs.shape[0] < batch_size:
                continue

            optimizer.zero_grad()
            inputs, targets = inputs.cuda(), targets.cuda()

            # forward
            logits, feat_mlp, centers = net(inputs)

            # 确保 feat_mlp 是二维
            if feat_mlp.dim() > 2:
                feat_mlp = feat_mlp.view(feat_mlp.size(0), -1)

            # 确保 centers 是二维
            if centers.dim() == 1:
                centers = centers.unsqueeze(0)  # [feat_dim] -> [1, feat_dim]
            elif centers.dim() > 2:
                centers = centers.view(centers.size(0), -1)
            # -------------------------------
            # 安全获取 batch 中每个样本对应的类别中心
            # -------------------------------
           
            # 调用改过的 ABCL 损失函数
            loss_ABCL = ABCL_loss(centers, feat_mlp, targets) 
            
            # 交叉熵 / 标签平滑损失
            loss_ce = LSLoss(logits, targets)

            # 总损失
            loss = loss_ce + args.beta * loss_ABCL.cuda()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 更新 top1 精度
            acc1 = accuracy(logits, targets, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

        scheduler.step()
        train_acc = top1.avg
        train_loss = train_loss / (idx + 1)
        print(
            'Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f}% '.format(
                lr_now, train_loss, train_acc
            )
        )

        ##### Evaluating model with test data every epoch
        with torch.no_grad():
            net.eval()
            eval_set = ImageFolder(
                root=join(data_dir, data_sets[-1]), transform=test_transform
            )
            eval_loader = DataLoader(
                eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            top1 = AverageMeter('Acc@1', ':6.2f ')
            for _, (inputs, targets) in enumerate(tqdm(eval_loader, ncols=80)):
                inputs, targets = inputs.cuda(), targets.cuda()
                x = net(inputs)
                acc1 = accuracy(x, targets, topk=(1,))
                top1.update(acc1[0].item(), batch_size)
            eval_acc = top1.avg
            print('{} | Acc: {:.3f}%'.format(data_sets[-1], eval_acc))

            ##### Logging
            with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
                file.write(
                    '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%\n'.format(
                        epoch, lr_now, train_loss, train_acc, eval_acc
                    )
                )

            ##### save model with highest acc
            if eval_acc > max_eval_acc:
                max_eval_acc = eval_acc
                torch.save(
                    net.state_dict(),
                    os.path.join(exp_dir, 'max_acc.pth'),
                    _use_new_zipfile_serialization=False,
                )

    ########################
    ##### 3 - Testing  #####
    ########################
    print('\n\n===== TESTING =====')
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a') as file:
        file.write('===== TESTING =====\n')

    ##### load best model
    net.load_state_dict(torch.load(join(exp_dir, 'max_acc.pth')))
    net.eval()

    for data_set in data_sets:
        top1 = AverageMeter('Acc@1', ':6.2f')
        testset = ImageFolder(
            root=os.path.join(data_dir, data_set), transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(tqdm(testloader, ncols=80)):
                inputs, targets = inputs.cuda(), targets.cuda()
                x = net(inputs)
                acc1 = accuracy(x, targets, topk=(1,))
                top1.update(acc1[0].item(), batch_size)
        test_acc = top1.avg
        print('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

        ##### logging
        with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
            file.write('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

        # file.write('Dataset {}\trecall:{:.2f}\n'.format(data_set, recall))
        # file.write('Dataset {}\tf1:{:.2f}\n'.format(data_set, f1))

    with open(
        os.path.join(exp_dir, 'acc_{}_{:.2f}'.format(data_set, test_acc)), 'a+'
    ) as file:
        # save accuracy as file name
        pass


