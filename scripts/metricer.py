from operator import index
import os
from re import L
import hydra
import tqdm
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from data.dataset import ValidDataset
from models import define_model
from metric.metrics import *

class Metricer:
    def __init__(self, cfg, gpu):
        cudnn.benchmark = True
        self.gpu = gpu
        self.scale = cfg.models.generator.scale
        self.save_path = cfg.valid.save_path
        os.makedirs(self.save_path, exist_ok=True)

        self._init_model(cfg.models)
        self._load_state_dict(cfg.models)
        self._init_dataset(cfg)
        self._init_metrics(cfg.valid.metrics)

        self.calc_metrics()

    def _init_model(self, model):
        self.generator, _ = define_model(model, self.gpu, False)
        self.generator.eval()

    def _load_state_dict(self, model):
        if model.generator.path:
            ckpt = torch.load(
                model.generator.path, map_location=lambda storage, loc: storage
            )
            if len(ckpt) == 3:
                self.generator.load_state_dict(ckpt["g"])
            else:
                self.generator.load_state_dict(ckpt)
    
    def _init_metrics(self, metric):
        metrics = []
        for m in metric:
            if m == 'psnr':
                metrics.append(PSNR())
            elif m == 'ssim':
                metrics.append(SSIM())
            elif m == 'lpips':
                metrics.append(LPIPS())
            elif m == 'erqa':
                metrics.append(ERQA())
        self.metrics = metrics
    
    def _init_dataset(self, cfg):
        num_workers = (
            8 * torch.cuda.device_count()
        )

        train_dataset = ValidDataset(cfg)

        self.dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def calc_metrics(self):
        scores = {}
        filenames = []
        to_tensor = transforms.ToTensor()

        for m in self.metrics:
            scores[m.name] = []

        for lrs, hr, lrnames in tqdm.tqdm(self.dataloader):
            lrnames = np.array(lrnames).flatten()
            hr = to_tensor(np.array(hr).squeeze(0)).unsqueeze(0).to(self.gpu)
            lrs = np.stack(lrs, 0)

            for lr in lrs:
                lr = to_tensor(lr.squeeze(0)).unsqueeze(0).to(self.gpu)

                with torch.no_grad():
                    preds = self.generator(lr)
                
                for m in self.metrics:
                    scores[m.name].append(m(preds, hr).item())

            filenames.extend(lrnames)

        df = pd.DataFrame.from_dict(scores, orient="columns")
        df.index = filenames
        df = df.sort_index()
        df.to_csv("Quantitative_Score.csv")

@hydra.main(config_path="../configs", config_name="valid.yaml")
def main(cfg):
    Metricer(cfg, 0)


if __name__ == "__main__":
    main()
