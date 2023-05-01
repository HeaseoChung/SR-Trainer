import cv2
import numpy as np
import random


def random_hflip(img):
    if random.random() < 0.5:
        img = np.fliplr(img)
    return img


def random_vflip(img):
    if random.random() < 0.5:
        img = np.flipud(img)
    return img


def random_roate(img):
    if random.random() < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def random_crop(hr, lr=None, crop_size=320, sf=2):
    max_x = hr.shape[1] - crop_size
    max_y = hr.shape[0] - crop_size

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    if not x % 2 == 0:
        x += 1

    if not y % 2 == 0:
        y += 1

    hr = hr[y : y + crop_size, x : x + crop_size]

    if lr is not None:
        y = y // sf
        x = x // sf
        lr = lr[y : y + crop_size // sf, x : x + crop_size // sf]

    return hr, lr
