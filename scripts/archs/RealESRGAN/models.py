from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.utils import spectral_norm
from archs.Utils.blocks import RRDB
from archs.Utils.utils import make_layer, pixel_unshuffle


class Generator(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.scale = cfg.scale
        num_in_ch = cfg.num_in_ch
        num_out_ch = cfg.num_out_ch
        num_feat = cfg.num_feat
        num_block = cfg.num_block
        num_grow_ch = cfg.num_grow_ch

        if self.scale == 2:
            num_in_ch = num_in_ch * 4
        elif self.scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


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
