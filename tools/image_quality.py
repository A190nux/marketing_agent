"""
image_quality.py
-----------------
Tool: check_image_quality

Purpose
    Decide whether an uploaded product photo needs super-resolution before
    it's used for an ad. Pure numpy/Pillow -- no heavy deps -- so it always
    runs, even before any GPU model is loaded.

Input
    image_path: str -- path to the uploaded image on disk

Output
    dict matching state.ImageQuality:
        {width, height, megapixels, blur_score, needs_enhancement}
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Below this we consider the photo "low resolution" for ad purposes.
MIN_MEGAPIXELS = 0.6          # ~ 800x750
# Below this Laplacian-variance the image is considered "blurry".
BLUR_VARIANCE_THRESHOLD = 80.0


def _laplacian_variance(gray: np.ndarray) -> float:
    """Cheap blur metric: variance of a 3x3 Laplacian response.
    Avoids an OpenCV dependency -- implemented directly with numpy.
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # same-size 'valid' convolution via simple padding + stride tricks
    padded = np.pad(gray, 1, mode="reflect")
    out = np.zeros_like(gray, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            w = kernel[dy, dx]
            if w == 0:
                continue
            out += w * padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
    return float(out.var())


def check_image_quality(image_path: str) -> dict:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        gray = np.asarray(img.convert("L"), dtype=np.float32)

    megapixels = round((width * height) / 1_000_000, 3)
    blur_score = round(_laplacian_variance(gray), 2)

    needs_enhancement = (megapixels < MIN_MEGAPIXELS) or (blur_score < BLUR_VARIANCE_THRESHOLD)

    return {
        "width": width,
        "height": height,
        "megapixels": megapixels,
        "blur_score": blur_score,
        "needs_enhancement": needs_enhancement,
    }
