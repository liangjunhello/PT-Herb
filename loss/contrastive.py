"""
增强版自适应平衡对比学习 (Enhanced Adaptive Balanced Contrastive Learning)
严格按照论文公式 (4.5),(4.6),(4.7),(4.8) 实现

Author: 基于Yonglong Tian的BalSCL
Date: 2025-12-03
说明: 完整实现论文中的自适应平衡对比学习框架，包含三个核心机制：
      1. 类别平均 (Class Averaging) - 公式 (4.5) 分母部分
      2. 类别补充 (Class Complement) - 公式 (4.5) 正样本集合和分母
      3. 自适应权重调整 (Adaptive Class Weight Adjustment) - 公式 (4.7),(4.8)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

_EPS = 1e-8  # 数值稳定性常量


class EnhancedBalSCL(nn.Module):
    """
    严格按论文公式 (4.5),(4.6),(4.7),(4.8) 实现的增强版自适应平衡对比学习

    核心方法说明：
    - compute_per_sample_lbcl(): 实现公式 (4.5)，计算每个样本的 L_{BCL}(x)
    - compute_class_avg_losses_from_lists(): 实现公式 (4.6)，计算每类平均损失 L_k^{avg}
    - update_adaptive_weights(): 实现公式 (4.7)，更新自适应权重 w_k = 1 + L_k^{avg} / L_max
    - forward(): 实现公式 (4.8)，计算最终损失 mean_i(w_{y_i} * L_{BCL}(x_i))
    """

    def __init__(self, cls_num_list: Optional[list] = None, temperature: float = 0.1, enable_adaptive: bool = False):
        """
        初始化增强版ABCL损失函数

        Args:
            cls_num_list: 全量类别样本数列表，用于确定 num_classes
            temperature: 温度参数 τ，控制相似度分布的尖锐程度
            enable_adaptive: 是否启用自适应权重机制（公式 4.7, 4.8）
        """
        super().__init__()
        self.temperature = float(temperature)  # 恢复温度参数
        self.cls_num_list = cls_num_list
        self.enable_adaptive = bool(enable_adaptive)

        # 确定类别数
        if cls_num_list is not None:
            self.num_classes = len(cls_num_list)
        else:
            self.num_classes = None

        # 创建自适应权重参数（requires_grad=False，不被优化器更新）
        if self.enable_adaptive and (self.num_classes is not None):
            self.adaptive_weights = nn.Parameter(torch.ones(self.num_classes), requires_grad=False)
        else:
            self.adaptive_weights = None

        # 用于追踪每类损失的字典（可选）
        self.class_loss_tracker = {i: [] for i in range(len(cls_num_list))} if cls_num_list else {}


    def compute_per_sample_lbcl(self, centers: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        严格按照公式 (4.5) 实现 L_{BCL} 损失计算

        公式 (4.5): 
        \mathcal{L}_{BCL} = -\frac{1}{|B_x|} \sum_{p \in ( B_x \setminus \{i\} ) \cup \{c_x\}} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{j \in X} \frac{1}{|B_j| + 1} \sum_{k \in B_j \cup \{c_j\}} \exp(z_i \cdot z_k / \tau)}
        """
        device = features.device
        B, D = features.shape  # B = batch size, D = feature dimension
        C = centers.shape[0]  # C = number of classes
        tau = float(self.temperature)

        # ===== 原始版本（有温度参数τ） - 恢复 =====
        # 计算样本间的相似度矩阵 [B, B] - z_i · z_k / τ
        sims = torch.matmul(features, features.t()) / self.temperature #表示所有样本之间的相似度。
        # features 的维度是 [B, D]，其中 B 是批次大小，D 是每个样本的特征维度
        exp_sims = torch.exp(sims)  # exp(z_i · z_k / τ)

        # 计算样本与类别中心的相似度 [B, C] - z_i · c_j / τ
        sims_centers = torch.matmul(features, centers.t()) / self.temperature #表示每个样本与每个类别中心之间的相似度。
        # centers 的维度是 [C, D]，其中 C 是类别数，D 是每个类别的特征维度（即类别的中心）
        exp_sims_centers = torch.exp(sims_centers)
        # ===========================================

        # 统计每个类别在当前 batch 的样本数 |B_j|
        counts = torch.zeros(C, device=device)
        unique, cnts = torch.unique(targets, return_counts=True)
        counts[unique] = cnts.float()

        # 存储每个样本的损失值
        per_sample_loss = torch.zeros(B, device=device)

        # 对每个样本计算公式 (4.5)
        for i in range(B):
            x = targets[i].item()  # 当前样本的类别索引

            # 获取同类别的其他样本索引，排除当前样本 i
            same_class_idx = (targets == x).nonzero(as_tuple=True)[0]
            same_class_idx = same_class_idx[same_class_idx != i]  # 排除自己

            # 正样本集合：同类样本 B_x ∪ {c_x}，这里 c_x 是类别中心
            if len(same_class_idx) > 0:
                pos_sims = torch.cat([sims[i, same_class_idx], sims_centers[i, x].unsqueeze(0)])
            else:
                pos_sims = sims_centers[i, x].unsqueeze(0)  # 如果没有同类样本，仅依赖类别中心

            # 计算分母部分 Σ_j (1 / (|B_j| + 1)) Σ_{k∈B_j ∪ {c_j}} exp(sim(x_i, k))
            denom = 0.0
            for j in range(C):  # 现在 j 遍历所有类别，保证归一化考虑到所有类别
                idx_j = (targets == j).nonzero(as_tuple=True)[0]  # 获取属于类别 j 的样本索引
                exp_sum = exp_sims[i, idx_j].sum() + exp_sims_centers[i, j]  # 计算该类样本的相似度之和 + 类别中心相似度
                denom += exp_sum / (counts[j] + 1.0)  # 归一化（使用 |B_j|+1）

            # 数值稳定的损失计算：log(a/b) = log(a) - log(b)
            log_terms = torch.log(pos_sims.exp() + _EPS) - torch.log(denom + _EPS)
            per_sample_loss[i] = -log_terms.mean()  # -(1 / |B_x|) * Σ

        return per_sample_loss  # 返回每个样本的损失 [B]


    def compute_class_avg_losses_from_lists(self, per_sample_losses: torch.Tensor, targets: torch.Tensor) -> Dict[int, float]:
        """
        按公式 (4.6) 计算每个类别的平均损失 L_k^{avg}

        公式 (4.6): L_k^{avg} = (1/|C_k|) Σ_{x∈C_k} L_{BCL}(x)

        Args:
            per_sample_losses: [B] 每个样本的 L_BCL 损失（来自验证集）
            targets: [B] 对应的类别标签

        Returns:
            class_avg_losses: dict {k: L_k^{avg}} 每类的平均损失
        """
        assert per_sample_losses.dim() == 1 and per_sample_losses.size(0) == targets.size(0)

        # 确定类别数
        if self.cls_num_list is not None:
            num_classes = len(self.cls_num_list)
        else:
            num_classes = int(targets.max().item()) + 1

        # 初始化每类的损失总和和样本数
        sums = [0.0] * num_classes
        counts = [0] * num_classes

        # 遍历每个样本的损失和标签，累积每个类别的损失和样本数
        for loss_val, t in zip(per_sample_losses.tolist(), targets.tolist()):
            k = int(t) # 类别标签
            if 0 <= k < num_classes:
                sums[k] += float(loss_val)
                counts[k] += 1

        # 计算平均损失 
        avg = {}
        for k in range(num_classes):
            if counts[k] > 0:
                avg[k] = sums[k] / counts[k]  # 公式 (4.6): L_k^{avg} = (1/|C_k|) Σ_{x∈C_k} L_{BCL}(x)
            else:
                avg[k] = 0.0
        return avg
    
    def update_adaptive_weights(self, class_avg_losses: Dict[int, float]):
        """
        按公式 (4.7) 更新自适应权重:
            w_k = 1 + L_k^{avg} / L_max

        Args:
            class_avg_losses: dict {k: L_k^{avg}}，每类平均损失（来自公式 4.6）
        """
        if (not self.enable_adaptive) or (not class_avg_losses) or (self.adaptive_weights is None):
            return

        device = self.adaptive_weights.device
        num_classes = self.adaptive_weights.numel()

        # 构建损失张量，缺失类别用0填充
        losses = torch.tensor([float(class_avg_losses.get(i, 0.0)) for i in range(num_classes)],
                              dtype=torch.float32, device=device)

        Lmax = losses.max().item()
        if Lmax <= 0:
            with torch.no_grad():
                self.adaptive_weights.data.fill_(1.0)  # 全部置为1（无权重）
            return

        with torch.no_grad():
            weights = 1.0 + losses / (Lmax + _EPS)  # 公式 (4.7): w_k = 1 + L_k^{avg} / L_max
            self.adaptive_weights.data.copy_(weights)

    def forward(self, centers: torch.Tensor, features: torch.Tensor, targets: torch.Tensor, use_class_weights: bool = False) -> torch.Tensor:
        """
        计算最终的 ABCL 损失

        Args:
            centers: [C, D] 类别中心矩阵
            features: [B, D] 批次特征矩阵
            targets: [B] 类别标签
            use_class_weights: 是否使用自适应权重（公式 4.8）

        Returns:
            loss: 最终的标量损失值
        """
        # 计算每个样本的基础 L_BCL 损失（公式 4.5）
        per_sample_losses = self.compute_per_sample_lbcl(centers, features, targets)  # [B]

        # 应用自适应权重（公式 4.8）: L = mean_i(w_{y_i} * L_{BCL}(x_i))
        if use_class_weights and self.enable_adaptive and (self.adaptive_weights is not None):
            weights = self.adaptive_weights.to(per_sample_losses.device)
            sample_weights = weights[targets.long()]  # w_{y_i}
            weighted = per_sample_losses * sample_weights
            return weighted.mean()  # 使用加权后的损失总和
        else:
            # 不使用自适应权重时，返回平均损失
            return per_sample_losses.mean()  # 使用未加权的平均损失
    
    # 向后兼容的别名
BalSCL = EnhancedBalSCL