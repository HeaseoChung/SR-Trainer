import os
import torch
import torchvision.utils as vutils
import pandas as pd

from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from data import define_dataset
from data.utils import postprocess
from archs import define_model
from methods.metric import define_metrics
from methods import Base


class Valider(Base):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
        ### GPU Device setting
        self.gpu = gpu
        self.ngpus_per_node = torch.cuda.device_count()
        self.scale = cfg.models.generator.scale

        ### Common setting
        self.save_csv = cfg.valid.common.save_csv
        self.save_path = cfg.valid.common.ckpt_dir
        os.makedirs(self.save_path, exist_ok=True)
        self.seed = cfg.valid.common.seed

        ### Model setting
        self.generator = self._init_model(cfg.models.generator)
        self.generator = self._load_state_dict(
            cfg.models.generator.path, self.generator
        )

        self._init_dataset(cfg)
        self._init_metrics(cfg)
        self._run()

    def _init_model(self, cfg):
        return define_model(cfg, self.gpu).eval()

    def _load_state_dict(self, path, model):
        if path:
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
            if len(ckpt) == 3:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
        return model

    def _init_dataset(self, cfg):
        cfg.valid.dataset.common.sf = self.scale
        valid_dataset = define_dataset(
            cfg.valid.dataset.common, cfg.valid.dataset.valid
        )

        self.dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=cfg.valid.dataset.valid.batch_size,
            shuffle=None,
            num_workers=cfg.valid.dataset.valid.num_workers,
            pin_memory=True,
            sampler=None,
            drop_last=True,
        )

    def _init_metrics(self, cfg):
        self.metrics = define_metrics(cfg.valid)

    def _visualize(self, i, img1, img2, img3):
        results = torch.cat(
            (
                img1.detach(),
                img2.detach(),
                img3.detach(),
            ),
            3,
        )
        vutils.save_image(
            results, os.path.join(self.save_path, f"compare_{i}.png")
        )

    def _save_csv(self, scores):
        df = pd.DataFrame.from_dict(scores, orient="columns")
        df.to_csv(os.path.join(self.save_path, "Quantitative_Score.csv"))

    def _run(self):
        self.generator.eval()
        scores = {}

        for k in self.metrics.keys():
            scores[k] = []

        for i, (lr, hr) in enumerate(self.dataloader):
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            with torch.no_grad():
                preds = self.generator(lr)

            preds = postprocess(preds)
            hr = postprocess(hr)

            for k in self.metrics.keys():
                score = self.metrics[k](preds, hr, self.scale)
                print(f"score[{k}]: {score}")
                scores[k].append(score)

            lr = F.interpolate(lr, scale_factor=self.scale, mode="nearest")

        for k in self.metrics.keys():
            avg_score = sum(scores[k]) / len(scores[k])
            scores["avg_" + k] = avg_score
            print(f"avg_{k}: {avg_score}")

        if self.save_csv:
            self._save_csv(scores)
