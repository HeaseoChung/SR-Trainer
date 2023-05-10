import os
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import os
from methods.test import Tester
from tqdm import tqdm
from data.utils import (
    check_image_file,
    preprocess,
    postprocess,
)


class TRT_Image(Tester):
    def __init__(self, cfg, gpu):
        super().__init__(cfg, gpu)
        self.n_color = cfg.models.generator.n_colors
        self.input_size = (
            cfg.test.data.batch,
            cfg.test.data.channel,
            cfg.test.data.height,
            cfg.test.data.width,
        )
        self.output_size = (
            cfg.test.data.batch,
            cfg.test.data.channel,
            cfg.test.data.height * self.scale,
            cfg.test.data.width * self.scale,
        )

        with open(cfg.models.generator.path, "rb") as f:
            self.engine = trt.Runtime(
                trt.Logger(trt.Logger.ERROR)
            ).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.img_test()

    def img_test(self):
        h_input = cuda.pagelocked_empty(
            trt.volume(self.input_size), dtype=np.float32
        )
        d_input = cuda.mem_alloc(h_input.nbytes)
        h_output = cuda.pagelocked_empty(
            trt.volume(self.output_size), dtype=np.float32
        )
        d_output = cuda.mem_alloc(h_output.nbytes)

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

            lr = np.array(img, np.float32)
            lr = lr.transpose([2, 0, 1])
            lr /= 255.0

            start = time.time()
            cuda.memcpy_htod(d_input, lr.ravel())
            self.context.execute(
                batch_size=self.input_size[0],
                bindings=[int(d_input), int(d_output)],
            )
            cuda.memcpy_dtoh(h_output, d_output)
            print(f"time : {time.time() - start}")

            preds = h_output.reshape(self.output_size)

            for p in preds:
                p *= 255.0
                p = p.transpose([1, 2, 0])
                p = np.clip(p, 0.0, 255.0).astype(np.uint8)
                p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
                cv2.imwrite(
                    os.path.join(self.save_path, path.split("/")[-1]),
                    p,
                )
