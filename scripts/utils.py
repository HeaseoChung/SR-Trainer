import torch
import numpy as np


def check_image_file(filename: str):
    return any(
        filename.endswith(extension)
        for extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".BMP",
        ]
    )


def check_video_file(filename: str):
    return any(
        filename.endswith(extension)
        for extension in [
            "mp4",
            "m4v",
            "mkv",
            "webm",
            "mov",
            "avi",
            "wmv",
            "mpg",
            "flv",
            "m2t",
            "mxf",
            "MXF",
        ]
    )


def uint2single(img):
    return np.float32(img / 255.0)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.0).round())


def preprocess(img):
    # uInt8 -> float32로 변환
    x = img.astype(np.float32)
    x = x.transpose([2, 0, 1])
    # Normalize x 값
    x /= 255.0
    # 넘파이 x를 텐서로 변환
    x = torch.from_numpy(x)
    # x의 차원의 수 증가
    x = x.unsqueeze(0)
    # x 값 반환
    return x


def postprocess(tensor):
    x = tensor.mul(255.0).cpu().numpy().squeeze(0)
    x = np.array(x).transpose([1, 2, 0])
    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return x
