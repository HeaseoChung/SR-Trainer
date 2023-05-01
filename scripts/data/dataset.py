import os
import torchvision.transforms as transforms
import cv2
import hydra
import torch
import torchvision.utils as vutils

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from data.degradation import Degradation
from data.augmentation import random_roate, random_crop
from data.utils import load_image_file


class ImageDegradationDataset(Dataset):
    def __init__(self, common, dataset):
        self.data_pipeline = Degradation(common, dataset)

        self.hrfiles = load_image_file(dataset.hr_dir)

        self.len = len(self.hrfiles)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        hr = cv2.imread(self.hrfiles[index])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        lr, hr = self.data_pipeline.data_pipeline(hr)
        return self.to_tensor(lr), self.to_tensor(hr)

    def __len__(self):
        return self.len


class ImagePairDataset(Dataset):
    def __init__(self, common, dataset):
        self.sf = common.sf
        self.patch_size = dataset.patch_size
        self.rand_crop = False if self.patch_size == -1 else True

        self.hrfiles = load_image_file(dataset.hr_dir)
        self.lrfiles = load_image_file(dataset.lr_dir)

        self.hrfiles.sort()
        self.lrfiles.sort()

        assert len(self.hrfiles) == len(
            self.lrfiles
        ), "The length of HR and LR files are not matched"
        self.len = len(self.hrfiles)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        hr = cv2.imread(self.hrfiles[index])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        lr = cv2.imread(self.lrfiles[index])
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        if self.rand_crop:
            hr, lr = random_crop(hr, lr, self.patch_size, self.sf)

        return self.to_tensor(lr), self.to_tensor(hr)

    def __len__(self):
        return self.len


@hydra.main(config_path="../../configs/", config_name="train.yaml")
def main(cfg):
    os.makedirs(cfg.train.common.save_img_dir, exist_ok=True)

    cfg.train.dataset.common.sf = 2
    train_dataset = ImagePairDegradationDataset(
        cfg.train.dataset.common, cfg.train.dataset.train
    )
    train_num_workers = (
        cfg.train.dataset.train.num_workers * torch.cuda.device_count()
    )

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.dataset.train.batch_size,
        shuffle=None,
        num_workers=train_num_workers,
        pin_memory=True,
        sampler=None,
        drop_last=True,
    )
    for idx, (lr, hr) in enumerate(dataloader):
        vutils.save_image(
            lr, os.path.join(cfg.train.common.save_img_dir, f"lr_{idx}.png")
        )
        vutils.save_image(
            hr, os.path.join(cfg.train.common.save_img_dir, f"hr_{idx}.png")
        )


if __name__ == "__main__":
    main()
