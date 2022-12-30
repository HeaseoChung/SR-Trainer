import os
import torch
import torch.nn as nn
import torchvision.utils as vutils

from train import Trainer
from torch.nn import functional as F


class GAN(Trainer):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
        self.generator = self._init_model(cfg, "generator")
        self.discriminator = self._init_model(cfg, "discriminator")

        if cfg.train.ddp.distributed:
            self._init_distributed_data_parallel(cfg)

        self.g_optim = self._init_optim(cfg, self.generator)
        self.d_optim = self._init_optim(cfg, self.discriminator)

        self.generator, self.g_optim = self._load_state_dict(
            cfg.models.generator.path, self.generator, self.g_optim
        )

        self.discriminator, self.d_optim = self._load_state_dict(
            cfg.models.discriminator.path, self.discriminator, self.d_optim
        )

        if (
            cfg.models.generator.path == ""
            or cfg.models.discriminator.path == ""
        ):
            self.start_iters = 0

        self.g_scheduler = self._init_scheduler(cfg, self.g_optim)
        self.d_scheduler = self._init_scheduler(cfg, self.d_optim)

        self._init_loss(cfg, gpu)
        self._init_dataset(cfg)
        self.train()

    def train(self):
        self.generator.train()
        self.discriminator.train()

        def requires_grad(model, flag=True):
            for p in model.parameters():
                p.requires_grad = flag

        for i in range(self.start_iters, self.end_iters):
            lr, hr = next(self.dataloader)
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            d_loss = 0.0

            preds = self.generator(lr)
            real_pred = self.discriminator(hr)
            d_loss_real = self.loss_lists["GANLoss"](real_pred, True)

            fake_pred = self.discriminator(preds)
            d_loss_fake = self.loss_lists["GANLoss"](fake_pred, False)

            d_loss = (d_loss_real + d_loss_fake) / 2

            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            preds = self.generator(lr)
            fake_pred = self.discriminator(preds)

            g_loss = 0.0
            for t_loss in self.loss_lists.keys():
                if t_loss == "GANLoss":
                    g_loss += self.loss_lists[t_loss](fake_pred, True)
                else:
                    g_loss += self.loss_lists[t_loss](preds, hr)

            self.generator.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            self.g_scheduler.step()
            self.d_scheduler.step()

            results = torch.cat(
                (
                    hr.detach(),
                    F.interpolate(
                        lr, scale_factor=self.scale, mode="nearest"
                    ).detach(),
                    preds.detach(),
                ),
                2,
            )

            if self.gpu == 0:
                if i % self.save_img_every == 0:
                    vutils.save_image(
                        results, os.path.join(self.save_path, f"preds.png")
                    )

                if i % self.save_model_every == 0:
                    if isinstance(self.generator, nn.DataParallel):
                        g_state_dict = self.generator.module.state_dict()
                        d_state_dict = self.discriminator.module.state_dict()
                    else:
                        g_state_dict = self.generator.state_dict()
                        d_state_dict = self.discriminator.state_dict()

                    torch.save(
                        {
                            "model": g_state_dict,
                            "optimimizer": self.g_optim.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(
                            self.save_path, f"g_{str(i).zfill(6)}.pth"
                        ),
                    )

                    torch.save(
                        {
                            "model": d_state_dict,
                            "optimimizer": self.d_optim.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(
                            self.save_path, f"d_{str(i).zfill(6)}.pth"
                        ),
                    )
