import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnextv2_atto(num_classes=100, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], num_classes=num_classes, **kwargs)
    return model


def convnextv2_femto(num_classes=100, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], num_classes=num_classes, **kwargs)
    return model


def convnext_pico(num_classes=100, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], num_classes=num_classes, **kwargs)
    return model


def convnextv2_nano(num_classes=100, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], num_classes=num_classes, **kwargs)
    return model


def convnextv2_tiny(num_classes=100, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, **kwargs)
    return model


def convnextv2_base(num_classes=100, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes, **kwargs)
    return model


def convnextv2_large(num_classes=100, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], num_classes=num_classes, **kwargs)
    return model


def convnextv2_huge(num_classes=100, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], num_classes=num_classes, **kwargs)
    return model


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c3, c4, c5 = x

        p5 = self.conv1(c5)
        p4 = self.conv2(c4)
        p3 = self.conv3(c3)

        p4 = F.interpolate(p5, size=p4.shape[-2:]) + p4
        p3 = F.interpolate(p4, size=p3.shape[-2:]) + p3

        f3 = self.conv4(p3)
        f4 = self.conv5(p4)
        f5 = self.conv6(p5)

        return f3, f4, f5


checkpoint = torch.load('convnextv2_base_22k_384_ema.pt')
predict = checkpoint['model']
model = convnextv2_base(num_classes=1000)
model.load_state_dict(predict)


# print(model)
# print("model is loaded")

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1).expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        # x: shape (batch_size, channels, height, width)

        # Squeeze: Global Average Pooling
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)

        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Reshape y to match x and scale the input
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class PromptBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False)
        self.depthWise_conv = nn.Conv2d(dim * 4, dim * 4, kernel_size=5, groups=dim, padding=2, bias=False)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.se = SEBlock(dim)
        self.eca = eca_layer(dim)

    def forward(self, x):
        x = self.ca(x)
        out = self.conv1(x)
        out = self.depthWise_conv(out)
        out = self.conv2(out)
        # out = self.ca(out)
        # out = self.sa(out)
        out = self.se(out)
        # out = self.eca(out)

        return out


class Conv2Fpn(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = model
        self.fpn = FPN([1024, 512, 256], 128)
        self.norm = nn.LayerNorm(128, eps=1e-6)  # 第一层维度进行拼接
        self.head = nn.Linear(128,180)
        # self.head = nn.Linear(1024, 180)

    def forward_features(self, x):

        p1 = self.backbone.downsample_layers[0](x)
        for block in self.backbone.stages[0]:
            p1 = block(p1)

        p2 = self.backbone.downsample_layers[1](p1)
        for block in self.backbone.stages[1]:
            p2 = block(p2)

        p3 = self.backbone.downsample_layers[2](p2)
        for block in self.backbone.stages[2]:
            p3 = block(p3)

        p4 = self.backbone.downsample_layers[3](p3)
        for block in self.backbone.stages[3]:
            p4 = block(p4)

        f2, f3, f4 = self.fpn([p2, p3, p4])  # 较深的三个阶段的输出采用特征金字塔 作为全局特征

        features = F.interpolate(f2, size=p1.shape[-2:]) + p1  # p1作为局部特征 与全局特征进行融合

        return self.norm(features.mean([-2, -1]))
        # return self.backbone.norm(p4.mean([-2, -1]))

    def forward(self, x):
        out = self.forward_features(x)
        out = self.head(out)

        return out


class CompressBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()
        self.grn = GRN(dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.act(x)
        x = self.grn(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Conv2MultiHeadFpn(nn.Module):
    def __init__(self, dims=[128, 256, 512, 1024]):
        super().__init__()
        self.backbone = model
        self.middlehead = nn.Sequential(
            nn.Linear(128, 180),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )

        self.fpn = FPN([1024, 512, 256], 128)
        self.norm = nn.LayerNorm(128, eps=1e-6)  # 第一层维度进行拼接
        self.head = nn.Linear(128, 180)

        self.steps = nn.ModuleList()
        for i in range(3):
            step = nn.Sequential(
                CompressBlock(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.steps.append(step)

        self.cnorm = nn.LayerNorm(1024, eps=1e-6)
        self.cHead = nn.Linear(1024, 180)

    def forward_features(self, x):

        p1 = self.backbone.downsample_layers[0](x)
        for block in self.backbone.stages[0]:
            p1 = block(p1)

        p2 = self.backbone.downsample_layers[1](p1)
        for block in self.backbone.stages[1]:
            p2 = block(p2)

        p3 = self.backbone.downsample_layers[2](p2)
        for block in self.backbone.stages[2]:
            p3 = block(p3)

        p4 = self.backbone.downsample_layers[3](p3)
        for block in self.backbone.stages[3]:
            p4 = block(p4)

        f2, f3, f4 = self.fpn([p2, p3, p4])  # 较深的三个阶段的输出采用特征金字塔 作为全局特征

        features = F.interpolate(f2, size=p1.shape[-2:]) + p1  # p1作为局部特征 与全局特征进行融合

        #     辅助分类器            #
        # return self.norm(p1.mean([-2, -1])), self.norm(features.mean([-2, -1]))

        #        添加了由下至上的特征压缩模块                 #
        c1 = self.steps[0](features)
        c2 = self.steps[1](torch.cat((c1, p2), dim=-1))
        c3 = self.steps[2](torch.cat((c2, p3), dim=-1))
        cFeatures = torch.cat((c3, p4), dim=-1)
        return self.cnorm(cFeatures.mean([-2, -1]))

    def forward(self, x):
        # middleout, out = self.forward_features(x)
        # middleout = self.middlehead(middleout)
        # out = self.head(out)
        #
        # return middleout, out

        out = self.forward_features(x)
        out = self.cHead(out)
        return out


class Prompt_Fpn(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = model
        self.fpn = FPN([1024, 512, 256], 128)
        self.norm = nn.LayerNorm(128, eps=1e-6)  # 第一层维度进行拼接
        self.head = nn.Linear(128, 180)
        self.prompt_blocks = nn.ModuleList([PromptBlock(128), PromptBlock(256), PromptBlock(512), PromptBlock(1024)])
        self.beta = nn.Parameter(torch.Tensor(1))
        # nn.init.xavier_uniform_(self.beta)

        # 冻结除 self.prompt_blocks, self.head 和 self.beta 外的所有参数
        self.freeze_weights()

    def freeze_weights(self):
        # 冻结 backbone, fpn 和 norm 的权重
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.fpn.parameters():
            param.requires_grad = False

        for param in self.norm.parameters():
            param.requires_grad = False

    def forward_features(self, x):

        p1 = self.backbone.downsample_layers[0](x)
        for block in self.backbone.stages[0]:
            p1 = block(p1)
        # print(f'p1.shape:{p1.shape}\n')
        p1 = p1 + self.beta * self.prompt_blocks[0](p1)

        p2 = self.backbone.downsample_layers[1](p1)
        for block in self.backbone.stages[1]:
            p2 = block(p2)
        # print(f'p2.shape:{p2.shape}\n')
        p2 = p2 + self.beta * self.prompt_blocks[1](p2)

        p3 = self.backbone.downsample_layers[2](p2)
        for block in self.backbone.stages[2]:
            p3 = block(p3)
        # print(f'p3.shape:{p3.shape}\n')
        p3 = p3 + self.beta * self.prompt_blocks[2](p3)

        p4 = self.backbone.downsample_layers[3](p3)
        for block in self.backbone.stages[3]:
            p4 = block(p4)
        # print(f'p4.shape:{p4.shape}\n')
        p4 = p4 + self.beta * self.prompt_blocks[3](p4)

        f2, f3, f4 = self.fpn([p2, p3, p4])  # 较深的三个阶段的输出采用特征金字塔 作为全局特征

        features = F.interpolate(f2, size=p1.shape[-2:]) + p1  # p1作为局部特征 与全局特征进行融合

        return self.norm(features.mean([-2, -1]))
        # return self.backbone.norm(p4.mean([-2, -1]))

    def forward(self, x):
        out = self.forward_features(x)
        out = self.head(out)

        return out


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out


class CL_Prompt_Fpn(nn.Module):
    def __init__(self, pretrained=False):# <- 新增 pretrained 参数
        super().__init__()
        self.backbone = model
        if pretrained:
            checkpoint = torch.load('convnextv2_base_22k_384_ema.pt', map_location='cuda')
            self.backbone.load_state_dict(checkpoint['model'], strict=False)
        self.fpn = FPN([1024, 512, 256], 128)
        self.norm = nn.LayerNorm(128, eps=1e-6)  # 第一层维度进行拼接
        # self.head = nn.Linear(128,180)
        self.prompt_blocks = nn.ModuleList([PromptBlock(128), PromptBlock(256), PromptBlock(512), PromptBlock(1024)])
        self.beta = nn.Parameter(torch.Tensor(1))
        # nn.init.xavier_uniform_(self.beta)

        self.head = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                  nn.Linear(128, 180))

        self.fc = NormedLinear(128, 180)

        self.head_fc = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                     nn.Linear(128, 180))

        # 冻结除 self.prompt_blocks, self.head 和 self.beta 外的所有参数
        self.freeze_weights()

    def freeze_weights(self):
        # 冻结 backbone, fpn 和 norm 的权重
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.fpn.parameters():
            param.requires_grad = False

        for param in self.norm.parameters():
            param.requires_grad = False

    def forward_features(self, x):

        p1 = self.backbone.downsample_layers[0](x)
        for block in self.backbone.stages[0]:
            p1 = block(p1)
        # print(f'p1.shape:{p1.shape}\n')
        p1 = p1 + self.beta * self.prompt_blocks[0](p1)

        p2 = self.backbone.downsample_layers[1](p1)
        for block in self.backbone.stages[1]:
            p2 = block(p2)
        # print(f'p2.shape:{p2.shape}\n')
        p2 = p2 + self.beta * self.prompt_blocks[1](p2)

        p3 = self.backbone.downsample_layers[2](p2)
        for block in self.backbone.stages[2]:
            p3 = block(p3)
        # print(f'p3.shape:{p3.shape}\n')
        p3 = p3 + self.beta * self.prompt_blocks[2](p3)

        p4 = self.backbone.downsample_layers[3](p3)
        for block in self.backbone.stages[3]:
            p4 = block(p4)
        # print(f'p4.shape:{p4.shape}\n')
        p4 = p4 + self.beta * self.prompt_blocks[3](p4)

        f2, f3, f4 = self.fpn([p2, p3, p4])  # 较深的三个阶段的输出采用特征金字塔 作为全局特征

        features = F.interpolate(f2, size=p1.shape[-2:]) + p1  # p1作为局部特征 与全局特征进行融合

        return self.norm(features.mean([-2, -1]))
        # return self.backbone.norm(p4.mean([-2, -1]))

    def forward(self, x):
        out = self.forward_features(x)
        feat_mlp = F.normalize(self.head(out), dim=1)
        out = self.fc(out)
        # print(f'out.shape:{out.shape}\n')

        # print(f'\nshape:{self.head.weight.T.shape}')
        centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1)

        if self.training:
            return out, feat_mlp, centers_logits
        else:
            return out