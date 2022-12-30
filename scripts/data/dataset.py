import os
import torchvision.transforms as transforms
import cv2


from torch.utils.data import Dataset
from data.degradation import Degradation
from data.augmentation import random_roate, random_crop
from utils import check_image_file


class ImagePairDegradationDataset(Dataset):
    def __init__(self, cfg):

        self.data_pipeline = Degradation(cfg)

        self.hrfiles = [
            os.path.join(cfg.train.dataset.train.hr_dir, x)
            for x in os.listdir(cfg.train.dataset.train.hr_dir)
            if check_image_file(x)
        ]

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
    def __init__(self, cfg):
        self.sf = cfg.models.generator.scale
        self.patch_size = cfg.train.dataset.train.patch_size

        self.hrfiles = [
            os.path.join(cfg.train.dataset.train.hr_dir, x)
            for x in os.listdir(cfg.train.dataset.train.hr_dir)
            if check_image_file(x)
        ]
        self.lrfiles = [
            os.path.join(cfg.train.dataset.train.lr_dir, x)
            for x in os.listdir(cfg.train.dataset.train.lr_dir)
            if check_image_file(x)
        ]

        assert len(self.hrfiles) == len(
            self.lrfiles
        ), "The length of HR and LR files are not matched"
        self.len = len(self.hrfiles)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        hr = cv2.imread(self.hrfiles[index])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        lr = cv2.imread(self.lrfiles[index])
        lr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        hr, lr = random_crop(hr, lr, self.patch_size, self.sf)

        return self.to_tensor(lr), self.to_tensor(hr)

    def __len__(self):
        return self.len
