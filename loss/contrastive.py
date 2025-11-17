"""
Author: Yonglong Tian (yonglong@mit.edu) - 简化版
Date: 2025-10-22
说明: 适配训练脚本，logits = batch x 类中心
"""

import torch
import torch.nn as nn

class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        """
        cls_num_list: 每个类别的样本数列表，可选，不影响本实现
        temperature: 对比损失的温度参数
        """
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers, features, targets):
        """
        centers: [num_classes, feat_dim] 所有类别中心
        features: [batch_size, feat_dim] 当前 batch 的特征
        targets: [batch_size] 当前 batch 的类别标签 (0~num_classes-1)
        """
        device = features.device
        batch_size = features.size(0)
        num_classes = centers.size(0)

        # logits: batch x 类中心
        logits = torch.mm(features, centers.t())  # [batch_size, num_classes]
        logits = logits / self.temperature

        # mask: batch x 类中心
        mask = torch.zeros_like(logits, device=device)
        mask[torch.arange(batch_size), targets] = 1.0  # 只有正确类别位置为1

        # 对比损失
        exp_logits = torch.exp(logits)  # [batch_size, num_classes]
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))  # log_softmax
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)  # batch内平均

        loss = -mean_log_prob_pos.mean()
        return loss
