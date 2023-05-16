import os
import cv2
import torch
import numpy as np


def load_image_file(root_dir: str):
    path = []
    for root, _, files in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if check_image_file(file_name):
                    path.append(os.path.join(root, file_name))
    return path


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


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r, :]
    else:
        raise ValueError("Wrong img ndim: [{:d}].".format(img.ndim))
    return img


def sharpen(img):
    kernel_sharpening_9 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpening_3 = np.array(
        [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]
    )
    kernel_sharpening_3_25 = np.array(
        [[-0.25, -0.25, -0.25], [-0.25, 3, -0.25], [-0.25, -0.25, -0.25]]
    )
    kernel_sharpening_1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpening_3)
    return sharpened
