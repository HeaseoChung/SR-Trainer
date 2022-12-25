import functools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
from archs.Utils.blocks import RRDB
from archs.Utils.utils import make_layer, pixel_unshuffle


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


# --------------------------------------------
# PatchGAN discriminator
# If n_layers = 3, then the receptive field is 70x70
# --------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        n_layers = cfg.n_layers
        norm_type = cfg.norm_type
        num_in_ch = cfg.num_in_ch
        num_feat = cfg.num_feat

        if norm_type == "batch":
            norm_layer = functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
            use_sp_norm = False
            use_bias = False
        elif norm_type == "instance":
            norm_layer = functools.partial(
                nn.InstanceNorm2d, affine=False, track_running_stats=False
            )
            use_sp_norm = False
            use_bias = True
        elif norm_type == "spectral":
            norm_layer = None
            use_sp_norm = True
            use_bias = True
        elif norm_type == "batchspectral":
            norm_layer = functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            )
            use_sp_norm = True
            use_bias = False
        elif norm_type == "none":
            norm_layer = None
            use_sp_norm = False
            use_bias = True
        else:
            raise NotImplementedError(
                "normalization layer [%s] is not found" % norm_type
            )

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            self.use_spectral_norm(
                nn.Conv2d(
                    num_in_ch, num_feat, kernel_size=kw, stride=2, padding=padw
                ),
                use_sp_norm,
            ),
            nn.LeakyReLU(0.2),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                self.use_spectral_norm(
                    nn.Conv2d(
                        num_feat * nf_mult_prev,
                        num_feat * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    ),
                    use_sp_norm,
                ),
                norm_layer(num_feat * nf_mult) if norm_layer else None,
                nn.LeakyReLU(0.2),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            self.use_spectral_norm(
                nn.Conv2d(
                    num_feat * nf_mult_prev,
                    num_feat * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                ),
                use_sp_norm,
            ),
            norm_layer(num_feat * nf_mult) if norm_layer else None,
            nn.LeakyReLU(0.2),
        ]
        sequence += [
            self.use_spectral_norm(
                nn.Conv2d(
                    num_feat * nf_mult,
                    1,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                ),
                use_sp_norm,
            )
        ]

        sequence_new = []
        for n in range(len(sequence)):
            if sequence[n] is not None:
                sequence_new.append(sequence[n])

        self.model = nn.Sequential(*sequence_new)

    def use_spectral_norm(self, module, mode=False):
        if mode:
            return spectral_norm(module)
        return module

    def forward(self, input):
        return self.model(input)
