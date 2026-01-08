import torch
import torch.nn as nn
from functools import partial
import yaml

import argparse
import os
import random
import shutil
from os.path import join

# import numpy as np  # 不再需要numpy，使用内置机制
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
from loss.enhanced_contrastive import EnhancedBalSCL
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


# 自适应权重管理已内置在EnhancedBalSCL中，无需外部管理器


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
    help='exp note, append after exp folder',
)
parser.add_argument(
    '-a',
    '--amp',
    default=0,
    help='0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp',
)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--beta', default=0.35, type=float, help='ABCL loss weight')
parser.add_argument('--alpha', default=0.9, type=float, help='LS loss weight')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate for ABCL training')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--enable_adaptive', action='store_true', default=True, help='enable adaptive class weighting')
parser.add_argument('--update_freq', default=5, type=int, help='frequency of updating adaptive weights (epochs)')
args = parser.parse_args()


##### exp setting
seed = int(args.seed)
datasets_dir = args.dir
nb_epoch = 128  # 128 as default to suit scheduler
batch_size = int(args.batch_size)
num_workers = int(args.num_workers)
lr_begin = args.lr  # 使用命令行参数指定学习率
use_amp = int(args.amp)  # use amp to accelerate training

##### data settings
current_dir = os.getcwd()  # 获取当前目录的路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # 获取上级目录的绝对路径
data_dir = datasets_dir if datasets_dir else '/data/ChapterTwo/dataset/aircraft_41split_seed42'

data_sets = ['train', 'test']
num_classes = len(
    os.listdir(join(data_dir, data_sets[0]))
)  # get number of class via img folders automatically
exp_dir = '/data/ChapterTwo/result_OURS/dataset_different/{}'.format(args.note)  # the folder to save model
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


##### CUDA device setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


##### Random seed setting
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
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
#     param.requires_grad = True  # make parameters in model learnable


##### optimizer setting

# 获取每个类别样本数
cls_num_list = [0] * len(train_set.classes)
for _, label in train_set.samples:
    cls_num_list[label] += 1

# Initialize enhanced ABCL loss with adaptive weighting
ABCL_loss = EnhancedBalSCL(
    cls_num_list=cls_num_list,
    temperature=args.temp,
    enable_adaptive=args.enable_adaptive
).cuda()

# ABCL损失函数已内置自适应权重机制，无需外部weight_manager
weight_manager = None  # 不再需要独立的权重管理器

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

shutil.copyfile('train_conv_enhanced.py', exp_dir + '/train_conv_enhanced.py')
shutil.copyfile('LabelSmoothing.py', exp_dir + '/LabelSmoothing.py')
shutil.copyfile('conv2Fpn.py', exp_dir + '/conv2Fpn.py')

with open(os.path.join(exp_dir, 'train_log.csv'), 'w+') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc, Test_Acc, Test_F1, Test_Recall\n')
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

        # Reset epoch losses for adaptive weighting
        # 使用内置的ABCL自适应权重机制，无需重置

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

            # 调用增强的 ABCL 损失函数
            loss_ABCL = ABCL_loss(centers, feat_mlp, targets, use_class_weights=args.enable_adaptive)

            # 交叉熵 / 标签平滑损失
            loss_LS = LSLoss(logits, targets)

            # Track per-sample ABCL losses for adaptive weighting (内置机制)
            if args.enable_adaptive:
                # 使用ABCL内置的per-sample损失计算功能
                with torch.no_grad():
                    per_sample_losses = ABCL_loss.compute_per_sample_lbcl(centers, feat_mlp, targets)
                    # 收集每类损失用于后续更新自适应权重
                    class_avg_losses = ABCL_loss.compute_class_avg_losses_from_lists(per_sample_losses, targets)
                    # 临时存储用于epoch结束时的权重更新
                    if not hasattr(ABCL_loss, '_epoch_losses'):
                        ABCL_loss._epoch_losses = {}
                    for cls_idx, avg_loss in class_avg_losses.items():
                        if cls_idx not in ABCL_loss._epoch_losses:
                            ABCL_loss._epoch_losses[cls_idx] = []
                        ABCL_loss._epoch_losses[cls_idx].append(avg_loss)

            # 总损失
            loss = args.alpha*loss_LS + args.beta * loss_ABCL.cuda()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 更新 top1 精度
            acc1 = accuracy(logits, targets, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

        # Update adaptive weights based on epoch performance (使用内置机制)
        if args.enable_adaptive and hasattr(ABCL_loss, '_epoch_losses') and (epoch + 1) % args.update_freq == 0:
            # 计算epoch中每类损失的平均值
            class_avg_losses = {}
            for cls_idx, losses in ABCL_loss._epoch_losses.items():
                if losses:
                    class_avg_losses[cls_idx] = sum(losses) / len(losses)

            # 更新自适应权重
            ABCL_loss.update_adaptive_weights(class_avg_losses)
            print(f"Updated adaptive weights at epoch {epoch+1} for {len(class_avg_losses)} classes")

            # 清空epoch损失记录
            ABCL_loss._epoch_losses = {}

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
            top1 = AverageMeter('Acc@1', ':6.2f')
            all_preds = []
            all_targets = []

            for _, (inputs, targets) in enumerate(tqdm(eval_loader, ncols=80)):
                inputs, targets = inputs.cuda(), targets.cuda()
                x = net(inputs)
                acc1 = accuracy(x, targets, topk=(1,))
                top1.update(acc1[0].item(), batch_size)

                # 收集预测和真实标签用于F1和Recall计算
                _, predicted = torch.max(x, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            eval_acc = top1.avg

            # 计算F1-score和Recall
            eval_f1 = f1_score(all_targets, all_preds, average='weighted') * 100
            eval_recall = recall_score(all_targets, all_preds, average='weighted') * 100

            print('{} | Acc: {:.3f}% | F1: {:.3f}% | Recall: {:.3f}%'.format(data_sets[-1], eval_acc, eval_f1, eval_recall))

            ##### Logging
            with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
                file.write(
                    '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%, {:.3f}%, {:.3f}%\n'.format(
                        epoch, lr_now, train_loss, train_acc, eval_acc, eval_f1, eval_recall
                    )
                )
                file.flush()

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

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(tqdm(testloader, ncols=80)):
                inputs, targets = inputs.cuda(), targets.cuda()
                x = net(inputs)
                acc1 = accuracy(x, targets, topk=(1,))
                top1.update(acc1[0].item(), batch_size)

                # 收集预测和真实标签用于F1和Recall计算
                _, predicted = torch.max(x, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        test_acc = top1.avg

        # 计算F1-score和Recall
        test_f1 = f1_score(all_targets, all_preds, average='weighted') * 100
        test_recall = recall_score(all_targets, all_preds, average='weighted') * 100

        print('Dataset {}\tACC:{:.2f}\tF1:{:.2f}\tRecall:{:.2f}\n'.format(data_set, test_acc, test_f1, test_recall))

        ##### logging
        with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
            file.write('Dataset {}\tACC:{:.2f}\tF1:{:.2f}\tRecall:{:.2f}\n'.format(data_set, test_acc, test_f1, test_recall))
            file.flush()

        # file.write('Dataset {}\trecall:{:.2f}\n'.format(data_set, recall))
        # file.write('Dataset {}\tf1:{:.2f}\n'.format(data_set, f1))

    with open(
        os.path.join(exp_dir, 'acc_{}_{:.2f}'.format(data_set, test_acc)), 'a+'
    ) as file:
        # save accuracy as file name
        pass

    # 训练结束后生成最高指标文件
    try:
        csv_path = os.path.join(exp_dir, 'train_log.csv')
        if os.path.exists(csv_path):
            print("正在生成最高指标文件...")

            # 读取CSV文件并找到最高指标
            import pandas as pd

            # 先读取CSV查看内容
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                print(f"CSV文件共有 {len(lines)} 行")
                if len(lines) > 1:
                    print(f"CSV内容示例: {lines[1].strip()}")

            df = pd.read_csv(csv_path)
            print(f"CSV列名: {list(df.columns)}")
            print(f"CSV数据行数: {len(df)}")

            # 检查是否存在测试数据行（不是TESTING部分）
            if len(df) > 0:
                # 查找数值格式的Test_Acc列
                if 'Test_Acc' in df.columns:
                    # 去掉百分号并转换为数值
                    df['Test_Acc_clean'] = df['Test_Acc'].astype(str).str.replace('%', '').astype(float)
                    df['Test_F1_clean'] = df['Test_F1'].astype(str).str.replace('%', '').astype(float)
                    df['Test_Recall_clean'] = df['Test_Recall'].astype(str).str.replace('%', '').astype(float)

                    # 过滤掉无效数据
                    valid_data = df[df['Test_Acc_clean'] > 0]
                    print(f"有效测试数据行数: {len(valid_data)}")

                    if len(valid_data) > 0:
                        # 找到最高准确率对应的epoch的指标
                        max_acc_row = valid_data.loc[valid_data['Test_Acc_clean'].idxmax()]
                        best_acc = max_acc_row['Test_Acc_clean']
                        best_f1 = max_acc_row['Test_F1_clean']
                        best_recall = max_acc_row['Test_Recall_clean']
                        best_epoch = int(max_acc_row['Epoch'])

                        # 生成指标文件
                        result_file = os.path.join(exp_dir, f'acc_{best_acc:.3f}_f1_{best_f1:.3f}_recall_{best_recall:.3f}')

                        with open(result_file, 'w') as f:
                            mode_str = "Adaptive" if args.enable_adaptive else "Standard"
                            f.write(f'Enhanced ABCL {mode_str} {dataset_name} 最高测试指标\n')
                            f.write(f'Test Acc: {best_acc:.3f}%\n')
                            f.write(f'Test F1: {best_f1:.3f}%\n')
                            f.write(f'Test Recall: {best_recall:.3f}%\n')
                            f.write(f'Epoch: {best_epoch}\n')

                        print(f'最高指标文件已生成: {result_file}')
                        print(f'最高准确率: {best_acc:.3f}% (Epoch {best_epoch})')
                    else:
                        print("没有找到有效的测试数据")
                else:
                    print("CSV文件中没有Test_Acc列")
            else:
                print("CSV文件为空")
        else:
            print(f"警告: 未找到日志文件 {csv_path}")
    except Exception as e:
        print(f'生成指标文件时出错: {e}')
        import traceback
        traceback.print_exc()