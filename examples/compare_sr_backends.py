"""
compare_sr_backends.py
-----------------------
Standalone comparison of the SR backends, independent of the LangGraph
agent/UI. Runs FOUR things side by side:

    1. real_esrgan            -- general-purpose trained model
    2. mri_model               -- your trained model, TILED (default, what
                                   the app actually uses -- see below)
    3. mri_model_whole_image   -- your trained model, run on the whole
                                   image in one shot (NOT tiled, NOT used
                                   by the app -- comparison only)
    4. bicubic                 -- plain interpolation, no model at all

Why tiling: confirmed via --native-scale (see git history / README) that
mri_model performs much better near its actual 64x64 training crop size
than run on a full-size photo directly (+222% Laplacian sharpness vs.
bicubic at native scale, vs. +5% -- indistinguishable -- at full size). The
app's default MRIModelBackend now tiles the image into overlapping 64x64
patches to run inference at that scale even on full-size uploads. This
script's mri_model_whole_image entry is the OLD non-tiled behavior, kept
around specifically so you can see the difference tiling makes -- it's
intentionally not reachable from the Streamlit app.

Why bicubic matters here specifically: RRDBNet's own architecture (see
mri_sr_model.py) has a bicubic term baked directly into its skip connection
(`out = out + F.interpolate(x, scale_factor=4, mode='bicubic')`). That means
it's a real, checkable possibility that a given checkpoint's *trained*
residual branch has learned to contribute little, and most of what you're
seeing from mri_model is actually just that skip connection -- which isn't
bad (it's a sensible inductive bias, and it guarantees the model is never
worse than bicubic), but it's worth knowing which one you're looking at.
This script computes a direct pixel-level diff between each model's output
and plain bicubic to quantify exactly that, rather than relying on eyeballing
images that can look similar at a glance for genuinely different reasons.

Unlike the agent's tool, this script FAILS LOUDLY if a backend can't load
its real weights, rather than silently falling back -- the whole point here
is to see the real model's behavior, not a stand-in.

Usage:
    python examples/compare_sr_backends.py path/to/product_photo.jpg
    python examples/compare_sr_backends.py path/to/product_photo.jpg --hr path/to/known_hr.jpg
    python examples/compare_sr_backends.py path/to/product_photo.jpg --native-scale

With --hr, also computes PSNR/SSIM against a genuine high-res reference for
all four (useful if you have a deliberately-downscaled test pair). Without
it, only the qualitative side-by-side and the diff-from-bicubic numbers are
produced -- there's no ground truth to score absolute quality against for
an arbitrary photo, but the diff-from-bicubic doesn't need one.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageDraw

from tools.super_resolution import RealESRGANBackend, MRIModelBackend, BicubicBackend  # noqa: E402
from tools.ad_image import _load_font  # noqa: E402  -- reuse the robust font fallback chain
from tools.image_quality import _laplacian_variance  # noqa: E402  -- reuse the sharpness metric

OUT_DIR = os.path.join(os.path.dirname(__file__), "sr_comparison_output")


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight global SSIM (not windowed) -- good enough for a rough
    comparison number here; use the training notebook's SSIMLoss for
    rigorous windowed SSIM if you need it."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def _diff_from_bicubic(name: str, im: Image.Image, bicubic_im: Image.Image) -> None:
    a = np.array(im.convert("RGB"), dtype=np.float64)
    b = np.array(bicubic_im.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        print(f"  {name} vs bicubic: sizes differ ({a.shape} vs {b.shape}), skipping diff")
        return
    mean_abs_diff = np.abs(a - b).mean()
    psnr_vs_bicubic = _psnr(a, b)
    print(f"  {name} vs bicubic: mean abs pixel diff = {mean_abs_diff:.2f} / 255, "
          f"PSNR = {psnr_vs_bicubic:.2f}dB")
    if psnr_vs_bicubic > 35:
        print(f"    -> very close to bicubic (high PSNR). The trained residual branch "
              f"is contributing little on this image -- not necessarily bad (never "
              f"worse than the baseline), but worth knowing.")
    elif psnr_vs_bicubic > 25:
        print(f"    -> moderately different from bicubic -- the model is doing "
              f"something, but bicubic likely still dominates the visible result.")
    else:
        print(f"    -> substantially different from bicubic -- the trained residual "
              f"branch is doing real work here.")


def _label(im: Image.Image, text: str) -> Image.Image:
    """Stamp a readable label under an image for the side-by-side grid."""
    font = _load_font(28)
    bar_h = 44
    labeled = Image.new("RGB", (im.width, im.height + bar_h), (20, 20, 20))
    labeled.paste(im, (0, 0))
    draw = ImageDraw.Draw(labeled)
    draw.text((10, im.height + 8), text, font=font, fill=(255, 255, 255))
    return labeled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to the product photo to test")
    parser.add_argument("--hr", default=None, help="optional ground-truth HR image for PSNR/SSIM")
    parser.add_argument("--allow-fallback", action="store_true",
                         help="don't error out if a backend falls back to Lanczos "
                              "(useful for testing the app's degraded-mode behavior)")
    parser.add_argument("--native-scale", action="store_true",
                         help="downsample the input to 64x64 first (mri_model's actual "
                              "training crop size) before running all three backends. "
                              "The model's fixed-pixel receptive field covers a much "
                              "larger relative fraction of a 64x64 image than a full-size "
                              "photo, so this isolates domain mismatch (MRI vs. photo "
                              "content) from scale mismatch (trained-crop-size vs. "
                              "full-photo-size) -- if the model's contribution shows up "
                              "here but not at full size, it's a scale effect, not just "
                              "domain mismatch.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_dir = os.path.join(OUT_DIR, "native_scale") if args.native_scale else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    with Image.open(args.image) as im:
        img = im.convert("RGB")

    if args.native_scale:
        img = img.resize((64, 64), Image.LANCZOS)
        print(f"--native-scale: downsampled input to 64x64 (mri_model's training crop size) first")
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")

    mri_whole = MRIModelBackend(tile=False)
    mri_whole.name = "mri_model_whole_image"  # distinct name -- separate output file, doesn't collide with tiled

    results = {}
    for backend in (RealESRGANBackend(), MRIModelBackend(), mri_whole, BicubicBackend()):
        print(f"\n--- {backend.name} ---")
        out = backend.enhance(img)
        if backend.used_fallback and not args.allow_fallback:
            print(f"ERROR: {backend.name} fell back to Lanczos: {backend.fallback_reason}")
            print("Fix the weights path / dependencies, or pass --allow-fallback to "
                  "compare the fallback behavior itself.")
            sys.exit(1)
        status = "REAL inference" if not backend.used_fallback else f"FALLBACK ({backend.fallback_reason})"
        print(f"Status: {status}")
        print(f"Output size: {out.size[0]}x{out.size[1]}")

        out_path = os.path.join(out_dir, f"{backend.name}.png")
        out.save(out_path)
        print(f"Saved: {out_path}")
        results[backend.name] = out

    print("\n=== mri_model: tiled (default, used by the app) vs. whole-image ===")
    a = np.array(results["mri_model"].convert("RGB"), dtype=np.float64)
    b = np.array(results["mri_model_whole_image"].convert("RGB"), dtype=np.float64)
    if a.shape == b.shape:
        print(f"  mean abs pixel diff: {np.abs(a - b).mean():.2f} / 255 "
              f"(0 would mean tiling made no difference at all)")
    else:
        print(f"  sizes differ ({a.shape} vs {b.shape}) -- both should be exactly 4x input, check for a bug")

    # the actual question this script exists to answer
    print("\n=== How much is each model contributing beyond plain bicubic? ===")
    print("(mean abs pixel diff / PSNR measure how MUCH changed, not whether it's\n"
          " structured sharpening vs. incoherent noise -- see sharpness section below\n"
          " for that distinction, which is usually what 'I can't tell visually' means.)")
    bicubic_im = results["bicubic"]
    for name in ("real_esrgan", "mri_model", "mri_model_whole_image"):
        _diff_from_bicubic(name, results[name], bicubic_im)

    print("\n=== Sharpness per output (Laplacian variance -- higher = more defined edges/texture) ===")
    sharpness = {}
    for name, im in results.items():
        gray = np.asarray(im.convert("L"), dtype=np.float32)
        sharpness[name] = _laplacian_variance(gray)
        print(f"  {name}: {sharpness[name]:.1f}")
    bicubic_sharpness = sharpness["bicubic"]
    for name in ("real_esrgan", "mri_model", "mri_model_whole_image"):
        delta = sharpness[name] - bicubic_sharpness
        pct = (delta / bicubic_sharpness * 100) if bicubic_sharpness else float("nan")
        if delta > bicubic_sharpness * 0.05:
            verdict = "meaningfully sharper than bicubic -- this is why it looks different"
        elif delta < -bicubic_sharpness * 0.05:
            verdict = "actually softer/smoother than bicubic"
        else:
            verdict = "essentially the same sharpness as bicubic -- consistent with 'can't tell visually'"
        print(f"  {name} vs bicubic: {delta:+.1f} ({pct:+.1f}%) -- {verdict}")

    # side-by-side comparison image, labeled
    labeled = {name: _label(im, name) for name, im in results.items()}
    w = max(r.size[0] for r in labeled.values())
    h = sum(r.size[1] for r in labeled.values()) + 40 * (len(labeled) - 1)
    combined = Image.new("RGB", (w, h), (20, 20, 20))
    y = 0
    for name, im in labeled.items():
        combined.paste(im, (0, y))
        y += im.size[1] + 40
    combined_path = os.path.join(out_dir, "side_by_side.png")
    combined.save(combined_path)
    print(f"\nSide-by-side comparison (labeled: real_esrgan / mri_model / mri_model_whole_image / bicubic) saved to: {combined_path}")

    if args.hr:
        with Image.open(args.hr) as hr_im:
            hr = np.array(hr_im.convert("RGB"))
        print("\n--- Metrics vs ground-truth HR ---")
        for name, im in results.items():
            pred = np.array(im.resize((hr.shape[1], hr.shape[0]), Image.LANCZOS))
            print(f"{name}: PSNR={_psnr(pred, hr):.2f}dB  SSIM(approx)={_ssim_simple(pred, hr):.4f}")


if __name__ == "__main__":
    main()