from __future__ import print_function
import os
import argparse
import random
import shutil
import numpy as np
from PIL import Image
import torch
import requests
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from src.models.tresnet_v2.tresnet_v2 import TResnetL_V2
from utilsCMAL import *
import torch.nn as nn
from basic_conv import BasicConv
from torch.utils.data import DataLoader  # 确保导入DataLoader
from torchvision import datasets, transforms  # 确保导入datasets
from randaugment import rand_augment_transform  # 导入 RandAugment
from os.path import join
from tqdm import tqdm

class Features(nn.Module):
    def __init__(self, net_layers_FeatureHead):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers_FeatureHead[0])
        self.net_layer_1 = nn.Sequential(*net_layers_FeatureHead[1])
        self.net_layer_2 = nn.Sequential(*net_layers_FeatureHead[2])
        self.net_layer_3 = nn.Sequential(*net_layers_FeatureHead[3])
        self.net_layer_4 = nn.Sequential(*net_layers_FeatureHead[4])
        self.net_layer_5 = nn.Sequential(*net_layers_FeatureHead[5])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x1 = self.net_layer_3(x)
        x2 = self.net_layer_4(x1)
        x3 = self.net_layer_5(x2)
        return x1, x2, x3


class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_classes):
        super().__init__()
        self.Features = Features(net_layers)
        self.max_pool1 = nn.MaxPool2d(kernel_size=46, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=23, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=12, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        map1 = x1_.clone().detach()
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)

        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        map2 = x2_.clone().detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        map3 = x3_.clone().detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3


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


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None, data_path='', seed=42):

    exp_dir = store_name
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # 设置随机种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    print('CUDA available:', use_cuda)

    print('==> Preparing data..')

    # 使用与原模型相同的数据变换
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

    # 数据加载 - 使用与原模型相同的路径和设置
    data_path = './data_from_web/Herb-LT_split42'  # 数据集路径
    num_workers = 12  # 与原模型一致

    trainset = datasets.ImageFolder(root=data_path + '/train', transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.ImageFolder(root=data_path + '/test', transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 自动检测类别数
    num_classes = len(trainset.classes)
    print(f'Number of classes: {num_classes}')

    model_params = {'num_classes': num_classes}

    # 加载 TResnetV2
    model = TResnetL_V2(model_params)
    weights_url = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/stanford_cars_tresnet-l-v2_96_27.pth'
    weights_path = "tresnet-l-v2.pth"

    if not os.path.exists(weights_path):
        print('Downloading weights...')
        r = requests.get(weights_url)
        with open(weights_path, "wb") as code:
            code.write(r.content)
    pretrained_weights = torch.load(weights_path)
    model.load_state_dict(pretrained_weights['model'])

    net_layers = list(model.children())
    net_layers = net_layers[0]
    net_layers = list(net_layers.children())
    net = Network_Wrapper(net_layers, num_classes)  # 使用自动检测的类别数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()

    # 使用与原模型相同的学习率
    lr_begin = 0.0017
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': lr_begin},
        {'params': net.conv_block1.parameters(), 'lr': lr_begin},
        {'params': net.classifier1.parameters(), 'lr': lr_begin},
        {'params': net.conv_block2.parameters(), 'lr': lr_begin},
        {'params': net.classifier2.parameters(), 'lr': lr_begin},
        {'params': net.conv_block3.parameters(), 'lr': lr_begin},
        {'params': net.classifier3.parameters(), 'lr': lr_begin},
        {'params': net.Features.parameters(), 'lr': lr_begin}
    ],
        momentum=0.9, weight_decay=5e-4)

    # 使用与原模型相同的学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128)

    max_val_acc = 0

    # 创建日志文件
    with open(os.path.join(exp_dir, 'train_log.csv'), 'w+') as file:
        file.write('Epoch, lr, Train_Loss, Train_Acc, Test_Acc\n')
        file.flush()

    for epoch in range(start_epoch, nb_epoch):
        print('\n===== Epoch: {} ====='.format(epoch))
        net.train()

        top1 = AverageMeter('Acc@1', ':6.2f')
        lr_now = optimizer.param_groups[0]['lr']
        train_loss = 0
        correct = 0
        total = 0
        idx = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, ncols=80)):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue

            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            # Train the experts from deep to shallow with data augmentation by multiple steps
            # e3
            inputs3 = inputs
            output_1, output_2, output_3, _, map1, map2, map3 = net(inputs3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()

            p1 = net.state_dict()['classifier3.1.weight']
            p2 = net.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output_3, p1, p2)
            inputs3_att = attention_im(inputs, att_map_3)

            p1 = net.state_dict()['classifier2.1.weight']
            p2 = net.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output_2, p1, p2)
            inputs2_att = attention_im(inputs, att_map_2)

            p1 = net.state_dict()['classifier1.1.weight']
            p2 = net.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output_1, p1, p2)
            inputs1_att = attention_im(inputs, att_map_1)
            inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)

            # e2
            optimizer.zero_grad()
            flag = torch.rand(1)
            if flag < (1 / 3):
                inputs2 = inputs3_att
            elif (1 / 3) <= flag < (2 / 3):
                inputs2 = inputs1_att
            elif flag >= (2 / 3):
                inputs2 = inputs

            _, output_2, _, _, _, map2, _ = net(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # e1
            optimizer.zero_grad()
            flag = torch.rand(1)
            if flag < (1 / 3):
                inputs1 = inputs3_att
            elif (1 / 3) <= flag < (2 / 3):
                inputs1 = inputs2_att
            elif flag >= (2 / 3):
                inputs1 = inputs

            output_1, _, _, _, map1, _, _ = net(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Train the experts and their concatenation with the overall attention region in one go
            optimizer.zero_grad()
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)
            concat_loss_ATT = CELoss(output_1_ATT, targets)+\
                            CELoss(output_2_ATT, targets)+\
                            CELoss(output_3_ATT, targets)+\
                            CELoss(output_concat_ATT, targets) * 2
            concat_loss_ATT.backward()
            optimizer.step()

            # Train the concatenation of the experts with the raw input
            optimizer.zero_grad()
            _, _, _, output_concat, _, _, _ = net(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            # 计算精度
            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # 计算总损失
            total_loss = loss1.item() + loss2.item() + loss3.item() + concat_loss.item()
            train_loss += total_loss

            # 更新精度
            acc1 = accuracy(output_concat, targets, topk=(1,))
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
            eval_set = datasets.ImageFolder(
                root=data_path + '/test', transform=test_transform
            )
            eval_loader = DataLoader(
                eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            top1 = AverageMeter('Acc@1', ':6.2f ')
            for _, (inputs, targets) in enumerate(tqdm(eval_loader, ncols=80)):
                inputs, targets = inputs.to(device), targets.to(device)
                output_1, output_2, output_3, output_concat, map1, map2, map3 = net(inputs)

                # 计算注意力图
                p1 = net.state_dict()['classifier3.1.weight']
                p2 = net.state_dict()['classifier3.4.weight']
                att_map_3 = map_generate(map3, output_3, p1, p2)

                p1 = net.state_dict()['classifier2.1.weight']
                p2 = net.state_dict()['classifier2.4.weight']
                att_map_2 = map_generate(map2, output_2, p1, p2)

                p1 = net.state_dict()['classifier1.1.weight']
                p2 = net.state_dict()['classifier1.4.weight']
                att_map_1 = map_generate(map1, output_1, p1, p2)

                inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
                output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)

                outputs_com2 = output_1 + output_2 + output_3 + output_concat
                outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

                acc1 = accuracy(outputs_com, targets, topk=(1,))
                top1.update(acc1[0].item(), batch_size)

            eval_acc = top1.avg
            print('Test | Acc: {:.3f}%'.format(eval_acc))

            ##### Logging
            with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
                file.write(
                    '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%\n'.format(
                        epoch, lr_now, train_loss, train_acc, eval_acc
                    )
                )

            ##### save model with highest acc
            if eval_acc > max_val_acc:
                max_val_acc = eval_acc
                torch.save(
                    net.state_dict(),
                    os.path.join(exp_dir, 'max_acc.pth'),
                    _use_new_zipfile_serialization=False,
                )

    print('\n\n===== TESTING =====')
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a') as file:
        file.write('===== TESTING =====\n')

    ##### load best model
    net.load_state_dict(torch.load(join(exp_dir, 'max_acc.pth')))
    net.eval()

    data_sets = ['train', 'test']
    for data_set in data_sets:
        top1 = AverageMeter('Acc@1', ':6.2f')
        testset = datasets.ImageFolder(
            root=os.path.join(data_path, data_set), transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(tqdm(testloader, ncols=80)):
                inputs, targets = inputs.to(device), targets.to(device)
                output_1, output_2, output_3, output_concat, map1, map2, map3 = net(inputs)

                # 计算注意力图
                p1 = net.state_dict()['classifier3.1.weight']
                p2 = net.state_dict()['classifier3.4.weight']
                att_map_3 = map_generate(map3, output_3, p1, p2)

                p1 = net.state_dict()['classifier2.1.weight']
                p2 = net.state_dict()['classifier2.4.weight']
                att_map_2 = map_generate(map2, output_2, p1, p2)

                p1 = net.state_dict()['classifier1.1.weight']
                p2 = net.state_dict()['classifier1.4.weight']
                att_map_1 = map_generate(map1, output_1, p1, p2)

                inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
                output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)

                outputs_com2 = output_1 + output_2 + output_3 + output_concat
                outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

                acc1 = accuracy(outputs_com, targets, topk=(1,))
                top1.update(acc1[0].item(), batch_size)

        test_acc = top1.avg
        print('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

        ##### logging
        with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
            file.write('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

        with open(
            os.path.join(exp_dir, 'acc_{}_{:.2f}'.format(data_set, test_acc)), 'a+'
        ) as file:
            # save accuracy as file name
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='', help='dataset dir')
    parser.add_argument('-b', '--batch_size', default=16, help='batch_size')
    parser.add_argument('-g', '--gpu', default='0', help='example: 0 or 1, to use different gpu')
    parser.add_argument('-w', '--num_workers', default=12, help='num_workers of dataloader')
    parser.add_argument('-s', '--seed', default=42, help='random seed')
    parser.add_argument('-n', '--note', default='', help='exp note, append after exp folder')
    args = parser.parse_args()

    # 实验设置
    seed = int(args.seed)
    datasets_dir = args.dir
    nb_epoch = 128  # 128 as default to suit scheduler
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)

    # 数据设置
    data_dir = './data_from_web/Herb-LT_split42'
    exp_dir = 'result_CMAL/{}{}'.format(datasets_dir, args.note)  # the folder to save model

    # 环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train(nb_epoch=nb_epoch,
          batch_size=batch_size,
          store_name=exp_dir,
          resume=False,
          start_epoch=0,
          model_path='',
          data_path=data_dir,
          seed=seed)