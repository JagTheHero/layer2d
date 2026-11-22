"""Real-ESRGAN x4plus anime 6B (the net texup.py uses in Test Signal's rig), for the canon and the references.

    up(img, k)      h x w x 3 (0..1) -> (k*h) x (k*w) x 3: the model's x4 in overlapping tiles, then Lanczos to k
    up_rgba(im, k)  straight RGBA: colour carried out under transparency first, alpha through the model as grey
"""
import os
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn

WEIGHTS = os.environ.get("REALESRGAN_WEIGHTS", "weights/RealESRGAN_x4plus_anime_6B.pth")
TILE, PAD = 256, 16


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        for i in range(5):
            setattr(self, f"conv{i + 1}", nn.Conv2d(nf + i * gc, gc if i < 4 else nf, 3, 1, 1))

    def forward(self, x):
        feats = [x]
        for i in range(5):
            y = getattr(self, f"conv{i + 1}")(torch.cat(feats, 1))
            feats.append(nn.functional.leaky_relu(y, 0.2) if i < 4 else y)
        return feats[-1] * 0.2 + x


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(), RDB(), RDB()

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """basicsr's RRDBNet, x4."""
    def __init__(self, blocks=6, nf=64):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB() for _ in range(blocks)])
        self.conv_body, self.conv_up1, self.conv_up2, self.conv_hr = (nn.Conv2d(nf, nf, 3, 1, 1) for _ in range(4))
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x):
        act = nn.functional.leaky_relu
        f = self.conv_first(x)
        f = f + self.conv_body(self.body(f))
        f = act(self.conv_up1(nn.functional.interpolate(f, scale_factor=2, mode="nearest")), 0.2)
        f = act(self.conv_up2(nn.functional.interpolate(f, scale_factor=2, mode="nearest")), 0.2)
        return self.conv_last(act(self.conv_hr(f), 0.2))


_NET = None


def net():
    global _NET
    if _NET is None:
        _NET = RRDBNet()
        _NET.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu")["params_ema"])
        _NET.eval()
    return _NET


@torch.no_grad()
def up4(img):
    h, w, _ = img.shape
    x = torch.from_numpy(np.ascontiguousarray(img, np.float32)).permute(2, 0, 1)[None]
    out = torch.zeros(1, 3, 4 * h, 4 * w)
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            a, b = max(0, y0 - PAD), max(0, x0 - PAD)
            t = net()(x[..., a:min(h, y0 + TILE + PAD), b:min(w, x0 + TILE + PAD)])
            out[..., 4 * y0:4 * min(h, y0 + TILE), 4 * x0:4 * min(w, x0 + TILE)] = \
                t[..., 4 * (y0 - a):4 * (y0 - a + min(TILE, h - y0)), 4 * (x0 - b):4 * (x0 - b + min(TILE, w - x0))]
    return out[0].clamp(0, 1).permute(1, 2, 0).numpy()


def resize(img, size):
    """float image (h x w x c, 0..1) to size (w, h), Lanczos per channel."""
    return np.stack([np.asarray(Image.fromarray(img[..., c].astype(np.float32), "F").resize(size, Image.LANCZOS))
                     for c in range(img.shape[2])], -1).clip(0, 1)


def up(img, k):
    h, w, _ = img.shape
    return resize(up4(img), (round(w * k), round(h * k)))


def up_rgba(im, k):
    rgb, a = im[..., :3], im[..., 3]
    iy, ix = ndimage.distance_transform_edt(a < 0.05, return_distances=False, return_indices=True)
    rgb2 = up(rgb[iy, ix], k)
    a2 = up(np.repeat(a[..., None], 3, -1), k).mean(-1)
    return np.concatenate([rgb2, a2[..., None]], -1)
