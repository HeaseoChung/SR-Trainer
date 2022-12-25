import torch
import torch.nn as nn
from torch.nn import init as init


class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(
                nn.Conv2d(
                    n_feats, n_feats, kernel_size=3, bias=True, padding=3 // 2
                )
            )
            if i == 0:
                m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        self.default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    @torch.no_grad()
    def default_init_weights(self, module_list, scale=1, bias_fill=0, **kwargs):
        if not isinstance(module_list, list):
            module_list = [module_list]
        for module in module_list:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, **kwargs)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, **kwargs)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x
