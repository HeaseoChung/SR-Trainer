"""
@article{zhang2022practical,
title={Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis},
author={Zhang, Kai and Li, Yawei and Liang, Jingyun and Cao, Jiezhang and Zhang, Yulun and Tang, Hao and Timofte, Radu and Van Gool, Luc},
journal={arXiv preprint},
year={2022}
}
This models.py From: https://github.com/cszn/scunet
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath


class WMSA(nn.Module):
    """Self-attention module in Swin Transformer"""

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(
            self.input_dim, 3 * self.input_dim, bias=True
        )

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1), self.n_heads
            )
        )

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(
                2 * window_size - 1, 2 * window_size - 1, self.n_heads
            )
            .transpose(1, 2)
            .transpose(0, 1)
        )

    def generate_mask(self, h, w, p, shift):
        """generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(
            h,
            w,
            p,
            p,
            p,
            p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(
            attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)"
        )
        return attn_mask

    def forward(self, x):
        """Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )
        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(
            x,
            "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")
        # Using Attn Mask to distinguish different subwindows.
        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows,
                w_windows,
                self.window_size,
                shift=self.window_size // 2,
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array(
                [
                    [i, j]
                    for i in range(self.window_size)
                    for j in range(self.window_size)
                ]
            )
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[
            :, relation[:, :, 0].long(), relation[:, :, 1].long()
        ]


class Block(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        head_dim,
        window_size,
        drop_path,
        type="W",
        input_resolution=None,
    ):
        """SwinTransformer Block"""
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"]
        self.type = type
        if input_resolution <= window_size:
            self.type = "W"

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(
        self,
        conv_dim,
        trans_dim,
        head_dim,
        window_size,
        drop_path,
        type="W",
        input_resolution=None,
    ):
        """SwinTransformer and Conv Block"""
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ["W", "SW"]
        if self.input_resolution <= self.window_size:
            self.type = "W"

        self.trans_block = Block(
            self.trans_dim,
            self.trans_dim,
            self.head_dim,
            self.window_size,
            self.drop_path,
            self.type,
            self.input_resolution,
        )
        self.conv1_1 = nn.Conv2d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            1,
            1,
            0,
            bias=True,
        )
        self.conv1_2 = nn.Conv2d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            1,
            1,
            0,
            bias=True,
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(
            self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1
        )
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange("b c h w -> b h w c")(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange("b h w c -> b c h w")(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.scale = cfg.scale
        self.in_nc = cfg.in_nc
        self.config = cfg.config
        self.dim = cfg.dim
        self.head_dim = cfg.head_dim
        self.window_size = cfg.window_size
        self.drop_path_rate = cfg.drop_path_rate
        self.input_resolution = cfg.input_resolution
        self.upsampler = cfg.upsampler

        # drop path rate for each layer
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, sum(self.config))
        ]

        self.m_head = [nn.Conv2d(self.in_nc, self.dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [
            ConvTransBlock(
                self.dim // 2,
                self.dim // 2,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                self.input_resolution,
            )
            for i in range(self.config[0])
        ] + [nn.Conv2d(self.dim, 2 * self.dim, 2, 2, 0, bias=False)]

        begin += self.config[0]
        self.m_down2 = [
            ConvTransBlock(
                self.dim,
                self.dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                self.input_resolution // 2,
            )
            for i in range(self.config[1])
        ] + [nn.Conv2d(2 * self.dim, 4 * self.dim, 2, 2, 0, bias=False)]

        begin += self.config[1]
        self.m_down3 = [
            ConvTransBlock(
                2 * self.dim,
                2 * self.dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                self.input_resolution // 4,
            )
            for i in range(self.config[2])
        ] + [nn.Conv2d(4 * self.dim, 8 * self.dim, 2, 2, 0, bias=False)]

        begin += self.config[2]
        self.m_body = [
            ConvTransBlock(
                4 * self.dim,
                4 * self.dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                self.input_resolution // 8,
            )
            for i in range(self.config[3])
        ]

        begin += self.config[3]
        self.m_up3 = [
            nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, 2, 2, 0, bias=False),
        ] + [
            ConvTransBlock(
                2 * self.dim,
                2 * self.dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                self.input_resolution // 4,
            )
            for i in range(self.config[4])
        ]

        begin += self.config[4]
        self.m_up2 = [
            nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 2, 2, 0, bias=False),
        ] + [
            ConvTransBlock(
                self.dim,
                self.dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                self.input_resolution // 2,
            )
            for i in range(self.config[5])
        ]

        begin += self.config[5]
        self.m_up1 = [
            nn.ConvTranspose2d(2 * self.dim, self.dim, 2, 2, 0, bias=False),
        ] + [
            ConvTransBlock(
                self.dim // 2,
                self.dim // 2,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                self.input_resolution,
            )
            for i in range(self.config[6])
        ]

        if self.upsampler == "nearest+conv":
            assert self.scale > 1
            self.m_tail = [
                nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False),
                nn.LeakyReLU(inplace=True),
            ]
            self.conv_up1 = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
            if self.scale == 4:
                self.conv_up2 = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
            self.conv_hr = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
            self.conv_last = nn.Conv2d(self.dim, self.in_nc, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.m_tail = [nn.Conv2d(self.dim, self.in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)
        # self.apply(self._init_weights)

    def forward(self, x0):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)

        if self.upsampler == "nearest+conv":
            x = self.lrelu(x)
            x = self.m_tail(x) + x1
            x = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(
                        x, scale_factor=2, mode="nearest"
                    )
                )
            )
            if self.scale == 4:
                x = self.lrelu(
                    self.conv_up2(
                        torch.nn.functional.interpolate(
                            x, scale_factor=2, mode="nearest"
                        )
                    )
                )

            x = self.conv_last(self.lrelu(self.conv_hr(x)))
            x = x[..., : h * self.scale, : w * self.scale]
        else:
            x = self.m_tail(x + x1)
            x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Discriminator(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)
    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.num_in_ch = cfg.num_in_ch
        self.num_feat = cfg.num_feat
        self.skip_connection = cfg.skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(
            self.num_in_ch, self.num_feat, kernel_size=3, stride=1, padding=1
        )
        # downsample
        self.conv1 = norm(
            nn.Conv2d(self.num_feat, self.num_feat * 2, 4, 2, 1, bias=False)
        )
        self.conv2 = norm(
            nn.Conv2d(self.num_feat * 2, self.num_feat * 4, 4, 2, 1, bias=False)
        )
        self.conv3 = norm(
            nn.Conv2d(self.num_feat * 4, self.num_feat * 8, 4, 2, 1, bias=False)
        )
        # upsample
        self.conv4 = norm(
            nn.Conv2d(self.num_feat * 8, self.num_feat * 4, 3, 1, 1, bias=False)
        )
        self.conv5 = norm(
            nn.Conv2d(self.num_feat * 4, self.num_feat * 2, 3, 1, 1, bias=False)
        )
        self.conv6 = norm(
            nn.Conv2d(self.num_feat * 2, self.num_feat, 3, 1, 1, bias=False)
        )
        # extra convolutions
        self.conv7 = norm(
            nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1, bias=False)
        )
        self.conv8 = norm(
            nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1, bias=False)
        )
        self.conv9 = nn.Conv2d(self.num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(
            x3, scale_factor=2, mode="bilinear", align_corners=False
        )
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(
            x4, scale_factor=2, mode="bilinear", align_corners=False
        )
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(
            x5, scale_factor=2, mode="bilinear", align_corners=False
        )
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


if __name__ == "__main__":

    # torch.cuda.empty_cache()
    net = Generator(cfg=0)

    x = torch.randn((2, 3, 64, 128))
    x = net(x)
    print(x.shape)
