from torch import nn as nn
from torch.nn import functional as F


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)

    p2d_h = (0, 0, 1, 0)
    p2d_w = (1, 0, 0, 0)

    if hh % 2 != 0:
        x = F.pad(x, p2d_h, "reflect")
    if hw % 2 != 0:
        x = F.pad(x, p2d_w, "reflect")
    h = x.shape[2] // scale
    w = x.shape[3] // scale

    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)
