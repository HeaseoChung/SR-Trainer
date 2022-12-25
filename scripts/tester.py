import os
import cv2
import hydra
import logging
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from tqdm import tqdm
from utils import check_image_file, check_video_file, preprocess, postprocess
from models import define_model


class Tester:
    def __init__(self, cfg, gpu):
        cudnn.benchmark = True
        self.gpu = gpu
        self.scale = cfg.models.generator.scale
        self.image_path = cfg.test.common.image_path
        self.save_path = cfg.test.common.save_path
        os.makedirs(self.save_path, exist_ok=True)

        self._init_model(cfg.models)
        self._load_state_dict(cfg.models)
        if cfg.test.common.type == "image":
            self.img_test()
        elif cfg.test.common.type == "video":
            self.video_test()
        else:
            raise ValueError("Test type should be image or video")

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
            h, w = img.shape[:2]
            img = cv2.resize(
                img,
                (w // self.scale, h // self.scale),
                interpolation=cv2.INTER_CUBIC,
            )
            lr = preprocess(img).to(self.gpu)

            with torch.no_grad():
                preds = self.generator(lr)

            cv2.imwrite(
                os.path.join(self.save_path, path.split("/")[-1]),
                postprocess(preds),
            )

        ### (train) SCUSR x2 total time to finish : 454.88920283317566
        ### (train) RealESRGAN x4 total time to finish : 922.6403729915619
        ### (train) RealESRGAN x2 total time to finish : 361.8168976306915
        ### (valid) SCUSR x2 total time to finish : 84.97539496421814
        ### (valid) RealESRGAN x2 total time to finish : 69.5964424610138

    def video_test(self):
        import ffmpeg

        videos = []

        if os.path.isdir(self.image_path):
            for img in os.listdir(self.image_path):
                if check_video_file(img):
                    videos.append(os.path.join(self.image_path, img))
        elif os.path.isfile(self.image_path):
            videos.append(self.image_path)
        else:
            raise ValueError("Neither a file or directory")

        for path in videos:

            target_file_name = os.path.join(self.save_path, path.split("/")[-1])
            streams = ffmpeg.probe(path, select_streams="v")["streams"][0]
            denominator, nominator = streams["r_frame_rate"].split("/")
            fps = float(denominator) / float(nominator)
            width = int(streams["width"])
            height = int(streams["height"])
            target_width = width * self.scale
            target_height = height * self.scale
            vcodec = streams["codec_name"]
            pix_fmt = streams["pix_fmt"]
            # color_range = streams["color_range"]
            # color_space = streams["color_space"]
            # color_transfer = streams["color_transfer"]
            # color_primaries = streams["color_primaries"]

            in_process = (
                ffmpeg.input(path)
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    loglevel="quiet",
                )
                .run_async(pipe_stdout=True)
            )

            out_process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s="{}x{}".format(target_width, target_height),
                    r=fps,
                )
                .output(
                    # ffmpeg.input(path).audio,
                    target_file_name,
                    pix_fmt=pix_fmt,
                    acodec="aac",
                    **{
                        "b:v": "50M",
                        # "color_range": color_range,
                        # "colorspace": color_space,
                        # "color_trc": color_transfer,
                        # "color_primaries": color_primaries,
                    },
                    vcodec=vcodec,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            while True:
                in_bytes = in_process.stdout.read(width * height * 3)
                if not in_bytes:
                    break
                in_frame = np.frombuffer(in_bytes, np.uint8).reshape(
                    [height, width, 3]
                )

                lr = preprocess(in_frame).to(self.gpu)

                with torch.no_grad():
                    preds = self.generator(lr)
                preds = postprocess(preds)
                out_process.stdin.write(preds.tobytes())

            in_process.stdout.close()
            out_process.stdin.close()
            out_process.wait()
            in_process.wait()


@hydra.main(config_path="../configs", config_name="test.yaml")
def main(cfg):
    Tester(cfg, 0)


if __name__ == "__main__":
    main()
