import cv2
import os
import torch
from methods.test import Tester
from tqdm import tqdm
from data.utils import (
    check_image_file,
    preprocess,
    postprocess,
)


class Image(Tester):
    def __init__(self, cfg, gpu):
        super().__init__(cfg, gpu)
        self.n_color = cfg.models.generator.n_colors
        self.generator = self._init_model(cfg.models.generator)
        self.generator = self._load_state_dict(
            cfg.models.generator.path, self.generator
        )
        self.img_test()

    def img_test(self):
        images = []

        if os.path.isdir(self.image_path):
            for img in os.listdir(self.image_path):
                if check_image_file(img):
                    images.append(os.path.join(self.image_path, img))
        elif os.path.isfile(self.image_path):
            images.append(self.image_path)
        else:
            raise ValueError("Neither a file or directory")

        for path in tqdm(images):
            img = cv2.imread(path)
            if self.n_color == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            lr = preprocess(img).to(self.gpu)

            with torch.no_grad():
                preds = self.generator(lr)

            preds = postprocess(preds)
            preds = cv2.cvtColor(preds, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                os.path.join(self.save_path, path.split("/")[-1]),
                preds,
            )
