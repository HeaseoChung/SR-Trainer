import os
import torch
import ffmpeg
import numpy as np

from methods.test import Tester
from tqdm import tqdm
from data.utils import (
    check_video_file,
    preprocess,
    postprocess,
)


class Video(Tester):
    def __init__(self, cfg, gpu):
        super().__init__(cfg, gpu)
        self.n_color = cfg.models.generator.n_colors
        self.generator = self._init_model(cfg.models.generator)
        self.generator = self._load_state_dict(
            cfg.models.generator.path, self.generator
        )
        self.video_test()

    def video_test(self):
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
