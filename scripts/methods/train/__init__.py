import os
import builtins
import torch
import torch.distributed as dist
import torchvision.utils as vutils
import wandb

from torch import nn
from torch.utils.data.dataloader import DataLoader
from archs import define_model
from data import define_dataset
from data.utils import postprocess, modcrop

from methods import Base
from methods.train.loss import define_loss
from methods.train.optim import define_optim
from methods.train.learning_rate_scheduler import define_LR_scheduler
from methods.metric import define_metrics


class Trainer(Base):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
        print(f"Train Method: {cfg.train.common.method} is going to be used")

        ### Common setting
        if cfg.train.ddp.distributed and self.gpu != 0:

            def print_pass(*args):
                pass

            builtins.print = print_pass

        self.save_path = cfg.train.common.ckpt_dir
        os.makedirs(self.save_path, exist_ok=True)
        self.start_iters = 0
        self.end_iters = cfg.train.common.iteration
        self.seed = cfg.train.common.seed
        self.use_wandb = cfg.train.common.use_wandb
        self.train_method = cfg.train.common.method
        if self.use_wandb and gpu == 0:
            wandb.init(project=f"{cfg.models.generator.name}")
            wandb.config.update(cfg)

        self.save_log_every = cfg.train.common.save_log_every
        self.save_img_every = cfg.train.common.save_img_every
        self.save_model_every = cfg.train.common.save_model_every

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
        self.valid_dataloader = None

        ### Metrics setting
        self.metrics = None
        self.avgerage = {}

    def _init_model(self, cfg_model, cfg_train):
        model = define_model(cfg_model, self.gpu)
        if cfg_train.train.ddp.distributed:
            print("Init distributed data parallel")
            model = self._init_distributed_data_parallel(cfg_train, model)
        return model

    def _load_state_dict(self, path, model, optim):
        if path:
            print(
                f"Load Checkpoint: Checkpoint is going to be used, Loading the checkpoint from : {path}"
            )
            ckpt = torch.load(
                path,
                map_location=lambda storage, loc: storage,
            )
            if len(ckpt) == 3:
                model.load_state_dict(ckpt["model"])
                if self.train_method == "NET":
                    optim.load_state_dict(ckpt["optimimizer"])
                    self.start_iters = ckpt["iteration"] + 1
            else:
                model.load_state_dict(ckpt)
        else:
            self.start_iters = 0
            print("Load Checkpoint: Checkpoint is not going to be used")

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

        train_dataset = define_dataset(
            cfg.models.generator.scale, cfg.train.dataset.train
        )
        valid_dataset = define_dataset(
            cfg.models.generator.scale, cfg.train.dataset.valid
        )

        train_num_workers = (
            cfg.train.dataset.train.num_workers * torch.cuda.device_count()
        )
        valid_num_workers = (
            cfg.train.dataset.valid.num_workers * torch.cuda.device_count()
        )

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=cfg.train.ddp.world_size,
                rank=self.gpu,
                shuffle=True,
            )
        else:
            train_sampler = None

        print(f"Number of train_dataset :{len(train_dataset)}")
        print(f"Number of valid_dataset :{len(valid_dataset)}")

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
        self.valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=cfg.train.dataset.valid.batch_size,
            shuffle=False,
            num_workers=valid_num_workers,
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

        cfg.train.dataset.train.batch_size = int(
            cfg.train.dataset.train.batch_size / self.ngpus_per_node
        )

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        print("Initialized the Distributed Data Parallel")
        return model

    def _init_metrics(self, cfg):
        self.metrics = define_metrics(cfg.train)

    def _valid(self, model):
        model.eval()
        scores = {}
        average = {}

        for k in self.metrics.keys():
            scores[k] = []

        for lr, hr in self.valid_dataloader:
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            with torch.no_grad():
                preds = model(lr)

            preds = postprocess(preds)
            hr = postprocess(hr)
            hr = modcrop(hr, self.scale)

            for k in self.metrics.keys():
                scores[k].append(self.metrics[k](preds, hr, self.scale))

        for k in self.metrics.keys():
            average[k] = sum(scores[k]) / len(scores[k])
        return average

    def _visualize(self, iter, imgs):
        results = None

        for img in imgs:
            if results == None:
                results = img
            else:
                results = torch.cat((results.detach(), img), 2)
        vutils.save_image(results, os.path.join(self.save_path, f"compare_{iter}.jpg"))

    def _save_model(self, name, iter, model, optim, average):
        state_dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        if len(self.avgerage) <= 0:
            for k in self.metrics.keys():
                self.avgerage[k] = 0

        for k in self.metrics.keys():
            if average[k] > self.avgerage[k]:
                self.avgerage[k] = average[k]
                torch.save(
                    state_dict,
                    os.path.join(self.save_path, f"best_{k}.pth"),
                )

        torch.save(
            {
                "model": state_dict,
                "optimimizer": optim.state_dict(),
                "iteration": iter,
            },
            os.path.join(self.save_path, f"{name}_{str(iter).zfill(6)}.pth"),
        )

    def _print(self, log):
        if self.use_wandb:
            wandb.log(log)
        else:
            print(log)
