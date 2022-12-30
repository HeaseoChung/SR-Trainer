import torch.nn as nn
from train.loss.l1_charbonnier_loss import L1_Charbonnier_loss
from train.loss.gan_loss import GANLoss
from train.loss.perceptual_loss import PerceptualLoss


def define_loss(cfg, gpu):
    loss_lists = {}

    for loss in cfg.train.loss.lists:
        if loss == "MAE":
            loss_lists[loss] = nn.L1Loss().to(gpu)
        elif loss == "Charbonnier":
            loss_lists[loss] = l1loss = L1_Charbonnier_loss().to(gpu)
        elif loss == "PerceptualLoss":
            loss_lists[loss] = PerceptualLoss(cfg.train.loss.PerceptualLoss).to(
                gpu
            )
        elif loss == "GANLoss":
            loss_lists[loss] = GANLoss(cfg.train.loss.GANLoss).to(gpu)
    print(f"Loss function: {loss_lists.keys()} is going to be used")
    return loss_lists
