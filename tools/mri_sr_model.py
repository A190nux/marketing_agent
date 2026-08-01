"""
mri_sr_model.py
----------------
The exact RRDBNet architecture used to train `best_sr_model.pt` (1-channel
T1 brain-MRI super-resolution, 64x64 -> 256x256, scale=4). Copied verbatim
from the training/eval notebook -- do NOT tweak layer shapes here without
retraining, or `load_state_dict` will fail (or silently load garbage if
shapes happen to coincide).

Architecture is fully convolutional (conv + PixelShuffle upsampling), so
despite being trained on fixed 64x64 -> 256x256 crops, it will run on any
input H/W at inference (output is exactly 4x each dimension).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.res_scale


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, res_scale)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, res_scale)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, res_scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale


class UpsampleBlock(nn.Module):
    def __init__(self, num_feat, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.pixel_shuffle(self.conv(x)))


def _initialize_weights(module, scale=0.1):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class RRDBNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, num_feat=64, num_block=8,
                 num_grow_ch=32, scale=4, init_scale=0.1):
        super().__init__()
        assert scale in (2, 4)
        self.scale = scale
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        n_up = 1 if scale == 2 else 2
        self.upsampler = nn.Sequential(*[UpsampleBlock(num_feat, scale=2) for _ in range(n_up)])
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        _initialize_weights(self, scale=init_scale)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.upsampler(feat)
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        out = out + base
        return torch.clamp(out, 0.0, 1.0)


def load_mri_sr_model(checkpoint_path: str, device: str = "cuda") -> tuple[RRDBNet, dict]:
    """Load `best_sr_model.pt` and return (model, checkpoint_meta).

    checkpoint_meta includes whatever else was saved alongside the weights
    (epoch, val_psnr, ...) -- handy for logging/debugging which checkpoint
    is actually loaded in the running app.
    """
    model = RRDBNet(in_ch=1, out_ch=1, num_feat=64, num_block=8, num_grow_ch=32, scale=4)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    # strip "module." prefix in case it was saved from a DataParallel-wrapped model
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta
