import os
import builtins
import torch
import torch.distributed as dist
import torchvision.utils as vutils

from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from data import define_dataset
from archs import define_model
from train.loss import define_loss
from train.optim import define_optim
from train.learning_rate_scheduler import define_LR_scheduler
from metric import define_metrics


class Trainer:
    def __init__(self, gpu, cfg):
        print(f"Train Method: {cfg.train.common.method} is going to be used")

        ### GPU Device setting
        self.gpu = gpu
        self.ngpus_per_node = torch.cuda.device_count()
        self.scale = cfg.models.generator.scale

        if cfg.train.ddp.distributed and self.gpu != 0:

            def print_pass(*args):
                pass

            builtins.print = print_pass

        ### Common setting
        self.save_path = cfg.train.common.ckpt_dir
        os.makedirs(self.save_path, exist_ok=True)
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
        self.train_dataloader = None
        self.test_dataloader = None

        ### Metrics setting
        self.metrics = None
        self.avgerage = {}

    def _init_model(self, cfg, model_type):
        return define_model(cfg.models, self.gpu, model_type)

    def _load_state_dict(self, path, model, optim):
        if path:
            print(
                f"State dictionary: Checkpoint is going to be used, Loading the checkpoint from : {path}"
            )
            ckpt = torch.load(
                path,
                map_location=lambda storage, loc: storage,
            )
            if len(ckpt) == 3:
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt["model"])
                optim.load_state_dict(ckpt["optimimizer"])
                self.start_iters = ckpt["iteration"] + 1
            else:
                model.load_state_dict(ckpt)
        else:
            print("State dictionary: Checkpoint is not going to be used")

        return model, optim

    def _init_loss(self, cfg, gpu):
        self.loss_lists = define_loss(cfg, gpu)

    def _init_optim(self, cfg, model):
        return define_optim(cfg, model)

    def _init_scheduler(self, cfg, model):
        return define_LR_scheduler(cfg, model, self.start_iters)

    def _init_dataset(self, cfg):
        def sample_data(loader):
            while True:
                for batch in loader:
                    yield batch

        cfg.train.dataset.common.sf = self.scale
        train_dataset = define_dataset(
            cfg.train.dataset.common, cfg.train.dataset.train
        )
        test_dataset = define_dataset(
            cfg.train.dataset.common, cfg.train.dataset.test
        )

        train_num_workers = (
            cfg.train.dataset.train.num_workers * torch.cuda.device_count()
        )
        test_num_workers = (
            cfg.train.dataset.test.num_workers * torch.cuda.device_count()
        )

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=cfg.train.ddp.world_size,
                rank=self.gpu,
            )
        else:
            train_sampler = None

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.train.dataset.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=train_num_workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )

        self.train_dataloader = sample_data(train_dataloader)
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.train.dataset.test.batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            pin_memory=True,
            sampler=None,
            drop_last=True,
        )

    def _init_distributed_data_parallel(self, cfg, model):

        dist.init_process_group(
            backend=cfg.train.ddp.dist_backend,
            init_method=cfg.train.ddp.dist_url,
            world_size=cfg.train.ddp.world_size,
            rank=self.gpu,
        )

        torch.cuda.set_device(self.gpu)
        model.to(self.gpu)

        cfg.train.dataset.batch_size = int(
            cfg.train.dataset.batch_size / self.ngpus_per_node
        )

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.gpu]
        )

        print("Initialized the Distributed Data Parallel")
        return model

    def _init_metrics(self, cfg):
        self.metrics = define_metrics(cfg)

    def _test(self, iter):
        self.generator.eval()
        scores = {}
        average = {}

        for m in self.metrics:
            scores[m.name] = []

        for lr, hr in self.test_dataloader:
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            with torch.no_grad():
                preds = self.generator(lr)

            for m in self.metrics:
                scores[m.name].append(m(preds, hr).item())

        for m in self.metrics:
            average[m.name] = sum(scores[m.name]) / len(scores[m.name])
        return average

    def _visualize(self, hr, lr, preds):
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

        vutils.save_image(results, os.path.join(self.save_path, f"compare.png"))

    def _save_model(self, name, iter, model, optim, average):
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        for m in self.metrics:
            if len(self.avgerage) <= 0:
                for m in self.metrics:
                    self.avgerage[m.name] = 0
            else:
                if average[m.name] > self.avgerage[m.name]:
                    self.avgerage[m.name] = average[m.name]
                    torch.save(
                        state_dict,
                        os.path.join(self.save_path, f"best_{m.name}.pth"),
                    )

        torch.save(
            {
                "model": state_dict,
                "optimimizer": optim.state_dict(),
                "iteration": iter,
            },
            os.path.join(self.save_path, f"{name}_{str(iter).zfill(6)}.pth"),
        )
