import os
import hydra
import builtins
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import torchvision.utils as vutils

import torch.multiprocessing as mp
import torch.distributed as dist

from data import define_dataset
from archs import define_model
from loss import define_loss
from optim import define_optim
from learning_rate_scheduler import define_LR_scheduler


class Trainer:
    def __init__(self, gpu, cfg) -> None:
        ### GPU Device setting
        self.gpu = gpu
        self.ngpus_per_node = torch.cuda.device_count()

        if cfg.train.ddp.distributed and self.gpu != 0:

            def print_pass(*args):
                pass

            builtins.print = print_pass

        ### Common setting
        self.save_path = cfg.train.common.ckpt_dir
        os.makedirs(self.save_path, exist_ok=True)

        self.gan_train = cfg.train.common.GAN
        self.scale = cfg.models.generator.scale
        self.start_iters = 0
        self.end_iters = cfg.train.common.iteration
        self.seed = cfg.train.common.seed
        self.use_wandb = cfg.train.common.use_wandb
        self.save_img_every = cfg.train.common.save_img_every
        self.save_model_every = cfg.train.common.save_model_every

        ### Model setting
        self.generator = None
        self.discriminator = None

        ### Optimizer setting
        self.g_optim = None
        self.d_optim = None

        ### Scheduler setting
        self.g_scheduler = None
        self.d_scheduler = None

        ### Loss setting
        self.l1loss = None
        self.perceptual_loss = None
        self.gan_loss = None

        ### Dataset setting
        self.distributed = cfg.train.ddp.distributed
        self.dataloader = None

        ### Initializer
        self._init_model(cfg)
        if cfg.train.ddp.distributed:
            self._init_distributed_data_parallel(cfg)
        self._init_optim(cfg)
        self._init_loss(cfg, gpu)
        self._load_state_dict(cfg)
        self._init_scheduler(cfg)
        self._init_dataset(cfg)

        ### Train
        self.train()

    def _init_model(self, cfg):
        self.generator, self.discriminator = define_model(
            cfg.models, self.gpu, self.gan_train
        )

    def _load_state_dict(self, cfg):
        if cfg.models.generator.path:
            print("Train the generator with checkpoint")
            print(f"Loading the checkpoint from : {cfg.models.generator.path}")
            ckpt = torch.load(
                cfg.models.generator.path,
                map_location=lambda storage, loc: storage,
            )
            if len(ckpt) == 3:
                if isinstance(cfg.models, nn.DataParallel):
                    self.generator.module.load_state_dict(ckpt["g"])
                else:
                    self.generator.load_state_dict(ckpt["g"])
                self.g_optim.load_state_dict(ckpt["g_optim"])
                self.start_iters = ckpt["iteration"] + 1
            else:
                self.generator.load_state_dict(ckpt)
        else:
            print("Train the generator without checkpoint")

        if self.gan_train:
            if cfg.models.discriminator.path:
                print("Train the discriminator with checkpoint")
                print(
                    f"Loading the checkpoint from : {cfg.models.discriminator.path}"
                )
                ckpt = torch.load(
                    cfg.models.discriminator.path,
                    map_location="cuda:{}".format(self.gpu),
                )
                if len(ckpt) == 3:
                    if isinstance(cfg.models, nn.DataParallel):
                        self.discriminator.module.load_state_dict(ckpt["d"])
                    else:
                        self.discriminator.load_state_dict(ckpt["d"])
                    self.d_optim.load_state_dict(ckpt["d_optim"])
                else:
                    self.discriminator.load_state_dict(ckpt)
            else:
                print("Train the discriminator without checkpoint")

        print("Initialized the checkpoints")

    def _init_loss(self, cfg, gpu):
        self.loss_lists = define_loss(cfg, gpu)

    def _init_optim(self, cfg):
        self.g_optim = define_optim(cfg, self.generator)

        if self.gan_train:
            self.d_optim = define_optim(cfg, self.discriminator)

    def _init_scheduler(self, cfg):
        self.g_scheduler = define_LR_scheduler(
            cfg, self.g_optim, self.start_iters
        )
        if self.gan_train:
            self.d_scheduler = define_LR_scheduler(
                cfg, self.d_optim, self.start_iters
            )

    def _init_dataset(self, cfg):
        def sample_data(loader):
            while True:
                for batch in loader:
                    yield batch

        cfg.train.dataset.train.num_workers = (
            cfg.train.dataset.train.num_workers * torch.cuda.device_count()
        )

        train_dataset = define_dataset(cfg)

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=cfg.train.ddp.world_size,
                rank=cfg.train.ddp.rank,
            )
        else:
            train_sampler = None

        dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.train.dataset.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=cfg.train.dataset.train.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
        self.dataloader = sample_data(dataloader)

    def _init_distributed_data_parallel(self, cfg):
        cfg.train.ddp.rank = cfg.train.ddp.nr * cfg.train.ddp.gpus + self.gpu

        dist.init_process_group(
            backend=cfg.train.ddp.dist_backend,
            init_method=cfg.train.ddp.dist_url,
            world_size=cfg.train.ddp.world_size,
            rank=cfg.train.ddp.rank,
        )

        torch.cuda.set_device(self.gpu)
        self.generator.to(self.gpu)

        cfg.train.dataset.batch_size = int(
            cfg.train.dataset.batch_size / self.ngpus_per_node
        )

        self.generator = torch.nn.parallel.DistributedDataParallel(
            self.generator, device_ids=[self.gpu]
        )

        if self.gan_train:
            self.discriminator.to(self.gpu)
            self.discriminator = torch.nn.parallel.DistributedDataParallel(
                self.discriminator, device_ids=[self.gpu]
            )

        print("Initialized the Distributed Data Parallel")

    def train_psnr(self):
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
                            "g": g_state_dict,
                            "g_optim": self.g_optim.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(self.save_path, f"{str(i).zfill(6)}.pth"),
                    )

    def train_gan(self):
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
                            "g": g_state_dict,
                            "g_optim": self.g_optim.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(
                            self.save_path, f"g_{str(i).zfill(6)}.pth"
                        ),
                    )

                    torch.save(
                        {
                            "d": d_state_dict,
                            "d_optim": self.d_optim.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(
                            self.save_path, f"d_{str(i).zfill(6)}.pth"
                        ),
                    )

    def train(self):
        if not self.gan_train:
            self.train_psnr()
        else:
            self.train_gan()


@hydra.main(config_path="../configs/", config_name="train.yaml")
def main(cfg):
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(cfg.train.common.seed)

    if torch.cuda.device_count() > 1:
        print("Train with multiple GPUs")
        cfg.train.ddp.distributed = True
        gpus = torch.cuda.device_count()
        cfg.train.ddp.world_size = gpus * cfg.train.ddp.nodes
        mp.spawn(
            Trainer,
            nprocs=gpus,
            args=(cfg,),
        )
        dist.destroy_process_group()
    else:
        print("Train with single GPUs")
        Trainer(0, cfg)


if __name__ == "__main__":
    main()
