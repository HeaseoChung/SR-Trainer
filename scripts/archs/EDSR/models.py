import torch.nn as nn
from archs.Utils.blocks import ResBlock


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        scale = cfg.scale
        num_in_ch = cfg.num_in_ch
        num_out_ch = cfg.num_out_ch
        num_feat = cfg.num_feat
        num_block = cfg.num_block
        res_scale = cfg.res_scale

        self.head = nn.Conv2d(
            num_in_ch, num_feat, kernel_size=3, padding=3 // 2
        )
        body = [ResBlock(num_feat, res_scale) for _ in range(num_block)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Conv2d(
                num_feat,
                num_feat * (scale**2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(scale),
            nn.ReLU(True),
            nn.Conv2d(num_feat, num_out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
