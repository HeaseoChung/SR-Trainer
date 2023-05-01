import torch.nn as nn
import torch.nn.functional as F
from archs.ECBSR.ecb import ECB


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.module_nums = cfg.num_block
        self.num_feat = cfg.num_feat
        self.scale = cfg.scale
        self.colors = cfg.n_colors
        self.with_idt = cfg.with_idt
        self.act_type = cfg.act_type
        self.backbone = None
        self.upsampler = None

        self.shallow = nn.Conv2d(
            self.colors, self.num_feat, kernel_size=3, padding=3 // 2
        )

        backbone = []

        for i in range(self.module_nums):
            backbone += [
                ECB(
                    self.num_feat,
                    self.num_feat,
                    depth_multiplier=2.0,
                    act_type=self.act_type,
                    with_idt=self.with_idt,
                )
            ]

        self.tail = ECB(
            self.num_feat,
            self.colors * (self.scale**2),
            depth_multiplier=2.0,
            act_type="linear",
            with_idt=self.with_idt,
        )

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x):
        x = self.shallow(x)
        y = self.backbone(x) + x
        y = self.upsampler(self.tail(y))
        return y
