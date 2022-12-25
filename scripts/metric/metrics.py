import lpips
import erqa
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class PSNR:
    """Peak Signal to Noise Ratio
    img1, img2 range [0, 1]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")

        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


class SSIM:
    """Structure Similarity
    img1, img2 range [0, 1]"""

    def __init__(self):
        self.name = "SSIM"
        self.window_size = 11
        self.channel = 3
        self.sigma = 1.5
        self.window = self.create_window()

    def create_window(self):
        torch.Tensor([exp(-(x - self.window_size//2)**2/float(2*self.sigma**2)) for x in range(self.window_size)])
        _1D_window = torch.Tensor([exp(-(x - self.window_size//2)**2/float(2*self.sigma**2)) for x in range(self.window_size)]).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return Variable(_2D_window.expand(self.channel, 1, self.window_size, self.window_size).contiguous()).cuda()

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        
        mu1 = F.conv2d(img1, self.window, padding = self.window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, self.window, padding = self.window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, self.window, padding = self.window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding = self.window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding = self.window_size//2, groups = self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

class LPIPS:
    """Learned Perceptual Image Patch Similarity
    img1, img2 range [0, 1]"""
    def __init__(self):
        self.name = "LPIPS"
        self.lpips = lpips.LPIPS(net="alex").cuda()

    def __call__(self, img1, img2):
        return self.lpips(img1, img2).squeeze()
    
class ERQA:
    """Edge Restoration Quality Assessment
    img1, img2 range [0, 255]"""
    def __init__(self):
        self.name = "ERQA"
        self.erqa = erqa.ERQA()

    def __call__(self, img1, img2):
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        img1 = (torch.transpose(img1, 0, 2).data.cpu().numpy() * 255.0).astype(np.uint8)
        img2 = (torch.transpose(img2, 0, 2).data.cpu().numpy() * 255.0).astype(np.uint8)
        return self.erqa(img1, img2)