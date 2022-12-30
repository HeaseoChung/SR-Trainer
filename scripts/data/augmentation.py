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
        if random.random() < 0.5:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def random_crop(hr, lr=None, crop_size=320, sf=2):
    max_x = hr.shape[1] - crop_size
    max_y = hr.shape[0] - crop_size

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    hr = hr[y : y + crop_size, x : x + crop_size]

    if lr is not None:
        lr = lr[y : y + crop_size, x : x + crop_size]
        lr = cv2.resize(
            lr,
            (
                crop_size // sf,
                crop_size // sf,
            ),
            interpolation=random.choice([1, 2, 3]),
        )

    return hr, lr
    