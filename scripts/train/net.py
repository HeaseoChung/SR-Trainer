import os
import torch
import torch.nn as nn
import torchvision.utils as vutils

from train import Trainer
from torch.nn import functional as F


class Net(Trainer):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
        self.generator = self._init_model(cfg, "generator")
        if cfg.train.ddp.distributed:
            self._init_distributed_data_parallel(cfg)
        self.g_optim = self._init_optim(cfg, self.generator)
        self.generator, self.g_optim = self._load_state_dict(
            cfg.models.generator.path, self.generator, self.g_optim
        )
        self.g_scheduler = self._init_scheduler(cfg, self.g_optim)

        self._init_loss(cfg, gpu)
        self._init_dataset(cfg)
        self.train()

    def train(self):
        self.generator.train()

        for i in range(self.start_iters, self.end_iters):
            lr, hr = next(self.dataloader)
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            preds = self.generator(lr)

            loss = 0
            for t_loss in self.loss_lists.keys():
                loss += self.loss_lists[t_loss](preds, hr)

            self.generator.zero_grad()
            loss.backward()
            self.g_optim.step()
            self.g_scheduler.step()

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
                    else:
                        g_state_dict = self.generator.state_dict()

                    torch.save(
                        {
                            "model": g_state_dict,
                            "optimimizer": self.g_optim.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(self.save_path, f"{str(i).zfill(6)}.pth"),
                    )
