"""
super_resolution.py
--------------------
Tool: super_resolve_image  (THE required tool for this assignment)

Purpose
    Enhance a low-quality product image so it's usable in an ad. Exposes a
    single `SRModel` interface with two swappable backends, selectable from
    the UI:

    - "real_esrgan": general-purpose photo super-resolution (Real-ESRGAN,
                     ~64MB weights -- comfortably fits an 8GB GPU alongside
                     the reasoning LLM on the second GPU). Uses tiled
                     inference so arbitrarily large product photos don't
                     OOM your 8GB card.
    - "mri_model":   the team's own trained RRDBNet (see mri_sr_model.py),
                     originally built for single-channel T1 brain-MRI slices
                     (64x64 -> 256x256, scale=4), reused here on RGB product
                     photos via a YCbCr Y-channel pass-through. Loads the
                     real `best_sr_model.pt` checkpoint. Included for
                     comparison / re-use; NOT recommended for product
                     photography due to domain mismatch -- see README
                     "Simulated / limited components" and "SR model
                     resolution behavior" below.

    A third backend, "bicubic", is also registered (see BicubicBackend) --
    plain interpolation, no model. It's not exposed in the Streamlit UI
    (there's no reason a manager would pick it over real_esrgan), but it's
    used by examples/compare_sr_backends.py as a baseline: RRDBNet's own
    architecture has a bicubic term in its skip connection, so this baseline
    is the direct way to check how much a given checkpoint's trained
    residual branch is actually contributing versus just reproducing that
    skip connection.

Input
    image_path: str
    backend: "real_esrgan" | "mri_model"

Output
    enhanced_image_path: str

On resolution / "target size"
    Both backends are fully convolutional and always upscale whatever size
    they're given by exactly 4x -- there's no fixed "target resolution".
    `mri_model` was only ever *trained* on 64x64 -> 256x256 crops, though;
    feeding it a much larger image (e.g. a 1200x900 photo) runs without
    error but pushes it well outside the distribution it learned, so
    quality (not correctness) degrades the larger/more-detailed the input
    is relative to that training regime. `real_esrgan` was trained on
    varied natural-image crops and is the one built to handle full-size
    photos.

On fallback visibility
    Both backends fall back to a deterministic Lanczos upscale if their
    real weights/libraries aren't available -- and BOTH now print exactly
    why to stderr, and expose `.used_fallback` / `.fallback_reason` on the
    backend instance (see get_backend_status() below). If real_esrgan and
    mri_model outputs look identical, check stderr / call
    get_backend_status() first: it usually means both silently fell back
    to the same Lanczos path, not that the app is "not hooked up".
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

TOOLS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TOOLS_DIR)  # marketing_assistant/
GENERATED_DIR = os.path.join(PROJECT_ROOT, "static", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

UPSCALE_FACTOR = 4


def _log(name: str, msg: str) -> None:
    print(f"[{name}] {msg}", file=sys.stderr)


_basicsr_patched = False


def _patch_basicsr_torchvision_compat() -> None:
    """basicsr (a Real-ESRGAN dependency) imports
    `torchvision.transforms.functional_tensor`, which was removed in
    torchvision>=0.17 -- this breaks `import realesrgan` on any reasonably
    current torch/torchvision install with a ModuleNotFoundError, even
    though the function it wants (`rgb_to_grayscale`) still exists, just
    moved to `torchvision.transforms.functional`.

    Rather than requiring a manual sed patch to the installed basicsr
    package (fragile: doesn't survive reinstalls, differs per venv/OS, easy
    to forget), inject a small shim module into sys.modules before basicsr
    imports it, so the import resolves transparently. Idempotent -- safe to
    call on every load attempt.
    """
    global _basicsr_patched
    if _basicsr_patched:
        return
    try:
        import torchvision.transforms.functional_tensor  # type: ignore  # noqa: F401
        # already importable (older torchvision, or already patched) -- nothing to do
    except ModuleNotFoundError:
        import types
        import torchvision.transforms.functional as _F  # type: ignore

        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = _F.rgb_to_grayscale  # the only symbol basicsr needs
        sys.modules["torchvision.transforms.functional_tensor"] = shim
        _log("real_esrgan", "applied basicsr/torchvision compatibility shim "
                             "(torchvision.transforms.functional_tensor)")
    _basicsr_patched = True


class SRModel(ABC):
    name: str
    used_fallback: bool = False
    fallback_reason: str | None = None

    @abstractmethod
    def enhance(self, image: Image.Image) -> Image.Image:
        ...


class BicubicBackend(SRModel):
    """Plain bicubic upscaling, no model involved -- the baseline every SR
    model is implicitly compared against. Useful for isolating how much a
    trained model is actually contributing: RRDBNet (mri_sr_model.py) has a
    bicubic term baked directly into its skip connection
    (`out = out + F.interpolate(x, scale_factor=4, mode='bicubic')`), so
    it's a real, checkable possibility that a given checkpoint's residual
    branch has learned to contribute little and most of the visible
    "enhancement" is actually just this. See compare_sr_backends.py, which
    reports a quantitative diff between mri_model and this backend for
    exactly that reason."""

    name = "bicubic"
    used_fallback = False   # there's no fallback state -- this IS the baseline
    fallback_reason = None
    weights_path = "(none -- no model)"

    def enhance(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        return image.resize((w * UPSCALE_FACTOR, h * UPSCALE_FACTOR), Image.BICUBIC)


class _FallbackUpscaleMixin:
    """Deterministic, dependency-free stand-in used when the real backend's
    weights/libraries aren't available in this environment."""

    def _fallback(self, image: Image.Image, reason: str) -> Image.Image:
        self.used_fallback = True
        self.fallback_reason = reason
        _log(self.name, f"using Lanczos fallback -- {reason}")
        w, h = image.size
        return image.resize((w * UPSCALE_FACTOR, h * UPSCALE_FACTOR), Image.LANCZOS)


class RealESRGANBackend(_FallbackUpscaleMixin, SRModel):
    """General-purpose photo super-resolution. Takes an RGB image directly."""

    name = "real_esrgan"

    # Hardcoded path to the downloaded weights -- update this if you move the
    # file. Resolved relative to the project root (not the current working
    # directory), so it works regardless of where you run `streamlit run` from.
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "RealESRGAN_x4plus.pth")

    def __init__(self, weights_path: str | None = None, device: str = "cuda"):
        self.weights_path = weights_path or self.WEIGHTS_PATH
        self.device = device
        self._model = None
        self.used_fallback = False
        self.fallback_reason = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not os.path.exists(self.weights_path):
            _log(self.name, f"weights not found at {self.weights_path}")
            self._model = False
            return self._model
        try:
            _patch_basicsr_torchvision_compat()
            # Real dependency: pip install realesrgan basicsr
            from realesrgan import RealESRGANer  # type: ignore
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore

            arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self._model = RealESRGANer(
                scale=4,
                model_path=self.weights_path,
                model=arch,
                device=self.device,
                # tile inference so large product photos don't OOM an 8GB GPU;
                # tile=0 (no tiling) is fine for small images, RealESRGANer
                # only tiles when the image actually needs it.
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )
            _log(self.name, f"loaded weights from {self.weights_path}")
        except Exception as e:
            _log(self.name, f"failed to load ({type(e).__name__}: {e}) -- check "
                             f"`pip install realesrgan basicsr` and the basicsr/"
                             f"torchvision compatibility patch in the README")
            self._model = False
        return self._model

    def enhance(self, image: Image.Image) -> Image.Image:
        model = self._load()
        if not model:
            return self._fallback(image, self.fallback_reason or "model failed to load, see stderr above")
        try:
            output, _ = model.enhance(np.array(image), outscale=UPSCALE_FACTOR)
            self.used_fallback = False
            self.fallback_reason = None
            return Image.fromarray(output)
        except Exception as e:
            return self._fallback(image, f"inference failed: {type(e).__name__}: {e}")


class MRIModelBackend(_FallbackUpscaleMixin, SRModel):
    """Adapter around the team's single-channel T1 brain-MRI SR model
    (RRDBNet, see mri_sr_model.py — architecture copied verbatim from the
    training notebook so `best_sr_model.pt` loads correctly).

    Trained on 64x64 -> 256x256 (scale=4) crops. Confirmed via
    examples/compare_sr_backends.py --native-scale that the model's trained
    residual branch does meaningfully sharpen beyond the bicubic skip
    connection AT that scale (+222% Laplacian sharpness vs. bicubic in one
    test), but is essentially indistinguishable from bicubic when run on a
    full-size photo directly (+5%) -- the model's fixed-pixel receptive
    field sees full-size photo content at a much finer relative scale than
    anything in its training crops, on top of the MRI-vs-photo domain gap.

    TILING (default on, see `tile`): rather than running the whole photo
    through the model at once, split it into overlapping 64x64 LR tiles
    (matching the training crop size), run each tile through the model at
    the scale it was actually trained on, and blend the overlapping regions
    back together. This lets a full-size photo benefit from the
    sharpening effect confirmed above, instead of defaulting to
    near-bicubic. Overlap uses simple averaging in the overlap region
    (not a fancy taper) -- keeps the implementation simple; visible seams
    are unlikely at 8px overlap but not mathematically impossible.

    Whole-image (non-tiled) mode is kept available via `tile=False` for
    comparison -- intentionally NOT exposed through get_backend() /
    the Streamlit app, only reachable by constructing MRIModelBackend(tile=False)
    directly, as examples/compare_sr_backends.py does.

    The MRI model only understands 1-channel input. To reuse it on a 3-channel
    color product photo without retraining anything:

        1. Convert RGB -> YCbCr.
        2. Run the MRI model on the Y (luminance) channel only (tiled, see above).
        3. Upsample Cb/Cr with plain bicubic (chroma detail is far less
           perceptually important than luminance detail, and tiling it
           wouldn't help since the model never touches Cb/Cr anyway).
        4. Merge Y' + Cb' + Cr' back into RGB.

    This is a legitimate, documented domain adapter -- but the underlying
    model was trained on medical imagery, so results on real photos may
    still show over-smoothing or MRI-like texture artifacts even with
    tiling fixing the scale mismatch. Kept here as a secondary, clearly
    labeled option; `real_esrgan` is the recommended default for product
    photography.
    """

    name = "mri_model"

    # Hardcoded path to your trained checkpoint -- update this if you move the
    # file. Resolved relative to the project root (not the current working
    # directory), so it works regardless of where you run `streamlit run` from.
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_sr_model.pt")
    DEVICE = "cuda"

    # Matches the training crop size -- see compare_sr_backends.py --native-scale
    # for why this specific size matters (it's where the model was confirmed to
    # actually contribute beyond bicubic).
    TILE_SIZE = 64
    TILE_OVERLAP = 8

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        tile: bool = True,
        tile_size: int | None = None,
        tile_overlap: int | None = None,
    ):
        self.weights_path = weights_path or self.WEIGHTS_PATH
        self.device = device or self.DEVICE
        self.tile = tile
        self.tile_size = tile_size or self.TILE_SIZE
        self.tile_overlap = tile_overlap or self.TILE_OVERLAP
        self._model = None
        self._meta = None
        self.used_fallback = False
        self.fallback_reason = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available() and self.device.startswith("cuda"):
                _log(self.name, "CUDA not available, using CPU (will be slow)")
                self.device = "cpu"
            if not self.weights_path or not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"MRI SR weights not found at {self.weights_path}")

            from .mri_sr_model import load_mri_sr_model

            self._model, self._meta = load_mri_sr_model(self.weights_path, device=self.device)
            _log(self.name, f"loaded checkpoint from {self.weights_path} "
                             f"(epoch={self._meta.get('epoch')}, val_psnr={self._meta.get('val_psnr')})")
        except Exception as e:
            _log(self.name, f"failed to load ({type(e).__name__}: {e})")
            self.fallback_reason = f"checkpoint load failed: {type(e).__name__}: {e}"
            self._model = False
        return self._model

    def _run_model(self, tile_np: np.ndarray, model) -> np.ndarray:
        """Run the model on a single HxW float32 (0-255) tile, return the
        4x-upscaled float32 (0-255) result."""
        import torch  # type: ignore

        with torch.no_grad():
            t = torch.from_numpy(tile_np).float().unsqueeze(0).unsqueeze(0) / 255.0
            t = t.to(self.device)
            out = model(t).clamp(0, 1).squeeze().cpu().numpy() * 255.0
        return out

    def _run_on_y_channel(self, y: np.ndarray, model) -> np.ndarray:
        """Whole-image (non-tiled) path -- runs the model on the full Y
        channel in one shot. This is what the model saw as "not the trained
        scale" in the compare_sr_backends.py finding; kept for comparison,
        not used by default."""
        return self._run_model(y, model)

    @staticmethod
    def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
        """Start positions for overlapping tiles covering [0, length), each
        exactly `tile` wide, with the last tile's end pinned to `length`
        (shifted left rather than padded) so every pixel is covered by at
        least one tile without needing to pad the image."""
        if length <= tile:
            return [0]
        stride = tile - overlap
        starts = list(range(0, length - tile + 1, stride))
        if starts[-1] != length - tile:
            starts.append(length - tile)
        return starts

    def _run_on_y_channel_tiled(self, y: np.ndarray, model) -> np.ndarray:
        """Split the Y channel into overlapping tile_size x tile_size tiles
        (matching the model's training crop size), run each through the
        model independently, and blend back together with simple averaging
        in the overlap regions. See class docstring for why this matters --
        the model performs much better at this scale than run whole-image."""
        h, w = y.shape
        tile = min(self.tile_size, h, w)
        overlap = min(self.tile_overlap, tile // 2) if tile > 1 else 0

        y_starts = self._tile_starts(h, tile, overlap)
        x_starts = self._tile_starts(w, tile, overlap)

        scale = UPSCALE_FACTOR
        out_h, out_w = h * scale, w * scale
        accum = np.zeros((out_h, out_w), dtype=np.float64)
        weight = np.zeros((out_h, out_w), dtype=np.float64)

        for ys in y_starts:
            for xs in x_starts:
                patch = y[ys:ys + tile, xs:xs + tile]
                sr_patch = self._run_model(patch, model)
                oy, ox = ys * scale, xs * scale
                oh, ow = sr_patch.shape
                accum[oy:oy + oh, ox:ox + ow] += sr_patch
                weight[oy:oy + oh, ox:ox + ow] += 1.0

        weight[weight == 0] = 1.0  # shouldn't happen given _tile_starts guarantees full coverage
        return accum / weight

    def enhance(self, image: Image.Image) -> Image.Image:
        model = self._load()
        if not model:
            return self._fallback(image, self.fallback_reason or "model failed to load, see stderr above")
        try:
            ycbcr = image.convert("YCbCr")
            y, cb, cr = ycbcr.split()
            y_np = np.array(y, dtype=np.float32)

            if self.tile:
                y_sr = self._run_on_y_channel_tiled(y_np, model)
            else:
                y_sr = self._run_on_y_channel(y_np, model)
            new_size = (y_sr.shape[1], y_sr.shape[0])

            cb_sr = cb.resize(new_size, Image.BICUBIC)
            cr_sr = cr.resize(new_size, Image.BICUBIC)
            y_sr_img = Image.fromarray(np.uint8(np.clip(y_sr, 0, 255)))

            merged = Image.merge("YCbCr", (y_sr_img, cb_sr, cr_sr))
            self.used_fallback = False
            self.fallback_reason = None
            return merged.convert("RGB")
        except Exception as e:
            return self._fallback(image, f"inference failed: {type(e).__name__}: {e}")


_BACKENDS: dict[str, SRModel] = {}


def get_backend(name: str) -> SRModel:
    if name not in _BACKENDS:
        if name == "real_esrgan":
            _BACKENDS[name] = RealESRGANBackend()
        elif name == "mri_model":
            _BACKENDS[name] = MRIModelBackend()
        elif name == "bicubic":
            _BACKENDS[name] = BicubicBackend()
        else:
            raise ValueError(f"Unknown SR backend: {name}")
    return _BACKENDS[name]


def get_backend_status(name: str) -> dict:
    """Introspection helper for the UI: was the last call to this backend
    real inference, or did it silently fall back? Call AFTER
    super_resolve_image() to get the status of that call."""
    backend = get_backend(name)
    return {
        "used_fallback": backend.used_fallback,
        "fallback_reason": backend.fallback_reason,
        "weights_path": backend.weights_path,
    }


def super_resolve_image(image_path: str, backend: str = "real_esrgan") -> str:
    """Tool entrypoint used by the LangGraph node."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        model = get_backend(backend)
        enhanced = model.enhance(img)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(GENERATED_DIR, f"{base}_enhanced_{backend}.png")
    enhanced.save(out_path)
    return out_path