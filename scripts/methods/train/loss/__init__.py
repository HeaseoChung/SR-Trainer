import torch.nn as nn
from methods.train.loss.l1_charbonnier_loss import Charbonnier_loss
from methods.train.loss.gan_loss import GANLoss
from methods.train.loss.perceptual_loss import PerceptualLoss
from methods.train.loss.pytorch_wavelets import DWTForward


def define_loss(cfg, gpu):
    loss_lists = {}

    for loss in cfg.train.loss.lists:
        if loss == "MAE":
            loss_lists[loss] = nn.L1Loss().to(gpu)
        elif loss == "Charbonnier":
            loss_lists[loss] = l1loss = Charbonnier_loss().to(gpu)
        elif loss == "PerceptualLoss":
            loss_lists[loss] = PerceptualLoss(cfg.train.loss.PerceptualLoss).to(
                gpu
            )
        elif loss == "GANLoss":
            loss_lists[loss] = GANLoss(cfg.train.loss.GANLoss).to(gpu)
        elif loss == "Wavelet":
            loss_lists[loss] = DWTForward(
                wave=cfg.train.loss.WaveletLoss.type,
                J=cfg.train.loss.WaveletLoss.level,
                mode=cfg.train.loss.WaveletLoss.pad,
            ).to(gpu)

    print(f"Loss function: {loss_lists.keys()} is going to be used")
    return loss_lists
