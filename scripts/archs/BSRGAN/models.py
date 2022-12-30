import torch.nn as nn
import torch.nn.functional as F

from archs.Utils.blocks import RRDB
from archs.Utils.utils import make_layer


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.sf = cfg.scale
        in_nc = cfg.num_in_ch
        out_nc = cfg.num_out_ch
        nf = cfg.num_feat
        nb = cfg.num_block
        gc = cfg.num_grow_ch

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB, nb, num_feat=nf, num_grow_ch=gc)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )
        if self.sf == 4:
            fea = self.lrelu(
                self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
