"""
ad_image.py
-----------
Tool: generate_ad_image

Purpose
    Turn the super-resolved product photo into a finished advertising image:
    a background/banner frame with the offer text overlaid on the photo.

    Default path is deterministic Pillow compositing -- fast, free, never
    fails at demo time. An optional Stable Diffusion 1.5 img2img "stylize"
    pass can be layered on top if a GPU + `diffusers` are available; this is
    a stretch feature, not the baseline, and silently skipped if unavailable.

Input
    enhanced_image_path: str
    offer_text: str
    product: str
    style: "img2img" | "composite" = "composite"

Output
    ad_image_path: str

Text sizing
    The banner auto-fits: text is wrapped based on actual measured pixel
    width (not a fixed character count, which reads badly across different
    offer lengths), and the banner height + font size adjust together so
    longer offers get more room rather than shrinking illegibly.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

CANVAS_SIZE = (1080, 1080)          # square post, Instagram/Facebook friendly
MARGIN = 48
MIN_BANNER_HEIGHT = 220
MAX_BANNER_HEIGHT = 460             # cap so the product photo always keeps most of the canvas
ACCENT_COLOR = (232, 93, 4)         # warm orange banner
TEXT_COLOR = (255, 255, 255)

# Hardcoded model id for the optional SD1.5 img2img "stylize" pass (see
# _try_img2img_stylize). Only used if style="img2img" is requested AND
# torch/diffusers/a CUDA GPU are available -- otherwise silently skipped.
SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def _load_font(size: int) -> ImageFont.ImageFont:
    # 1) common system-installed DejaVu paths (Linux)
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)

    # 2) matplotlib ships DejaVuSans-Bold.ttf as package data -- if it's
    # installed (common in ML/data-science environments) we get a real
    # scalable font without needing OS-level font packages.
    try:
        import matplotlib  # type: ignore
        mpl_font = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
        if os.path.exists(mpl_font):
            return ImageFont.truetype(mpl_font, size)
    except Exception:
        pass

    # 3) Pillow (>=10.1.0, already a hard requirement of this project) ships
    # its own scalable built-in font via load_default(size=...) -- unlike
    # the old size-less load_default(), this one actually scales, so it's a
    # genuinely usable fallback with ZERO external dependencies. This should
    # cover any environment where the app runs at all.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        pass  # Pillow <10.1.0 -- load_default() doesn't take a size arg yet

    print(f"[ad_image] no scalable font available (checked system paths, matplotlib, "
          f"and Pillow's built-in font) -- falling back to PIL's tiny fixed-size "
          f"default font. Upgrade Pillow to >=10.1.0 (`pip install -U pillow`) to fix this.")
    return ImageFont.load_default()


def _wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Greedy word-wrap based on actually-measured pixel width, not a fixed
    character count (character-count wrapping looks fine for some offers and
    badly cramped/oversized for others depending on font metrics)."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_banner_text(
    draw: ImageDraw.ImageDraw, offer_text: str, product: str, max_width: int
) -> tuple[list[str], ImageFont.ImageFont, ImageFont.ImageFont, int]:
    """Pick the largest title font size (within a sane range) whose wrapped
    offer text + product subtitle fits within MAX_BANNER_HEIGHT. Returns
    (wrapped_lines, title_font, sub_font, banner_height)."""
    for title_size in range(76, 27, -4):  # try large first, shrink only if needed
        title_font = _load_font(title_size)
        sub_font = _load_font(max(20, int(title_size * 0.42)))

        lines = _wrap_to_width(draw, offer_text, title_font, max_width)
        line_height = int(title_size * 1.25)
        title_block_h = line_height * len(lines)
        sub_block_h = int(sub_font.size * 1.4) if hasattr(sub_font, "size") else 30

        banner_height = MARGIN * 2 + title_block_h + sub_block_h + 10
        banner_height = max(MIN_BANNER_HEIGHT, min(banner_height, MAX_BANNER_HEIGHT))

        # accept this size if the content actually fits within the cap
        # (or if we're already at the smallest size we're willing to try)
        content_fits = (MARGIN * 2 + title_block_h + sub_block_h + 10) <= MAX_BANNER_HEIGHT
        if content_fits or title_size <= 32:
            return lines, title_font, sub_font, banner_height

    # unreachable in practice, but keep a safe default
    title_font = _load_font(32)
    sub_font = _load_font(20)
    lines = _wrap_to_width(draw, offer_text, title_font, max_width)
    return lines, title_font, sub_font, MAX_BANNER_HEIGHT


def _composite(enhanced: Image.Image, offer_text: str, product: str) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    max_text_width = CANVAS_SIZE[0] - 2 * MARGIN
    lines, title_font, sub_font, banner_height = _fit_banner_text(
        draw, offer_text, product, max_text_width
    )

    # fit product photo into the top portion, center-cropped
    photo_area_h = CANVAS_SIZE[1] - banner_height
    photo = enhanced.copy()
    src_w, src_h = photo.size
    target_ratio = CANVAS_SIZE[0] / photo_area_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_h = src_h
        new_w = int(src_h * target_ratio)
    else:
        new_w = src_w
        new_h = int(src_w / target_ratio)
    left = (src_w - new_w) // 2
    top = (src_h - new_h) // 2
    photo = photo.crop((left, top, left + new_w, top + new_h)).resize(
        (CANVAS_SIZE[0], photo_area_h), Image.LANCZOS
    )
    canvas.paste(photo, (0, 0))

    # bottom banner with offer text, auto-sized above
    draw.rectangle(
        [(0, photo_area_h), (CANVAS_SIZE[0], CANVAS_SIZE[1])], fill=ACCENT_COLOR
    )

    line_height = int(title_font.size * 1.25) if hasattr(title_font, "size") else 60
    y = photo_area_h + MARGIN
    for line in lines:
        draw.text((MARGIN, y), line, font=title_font, fill=TEXT_COLOR)
        y += line_height

    draw.text((MARGIN, CANVAS_SIZE[1] - MARGIN - (sub_font.size if hasattr(sub_font, "size") else 24)),
               product, font=sub_font, fill=TEXT_COLOR)
    return canvas


def _try_img2img_stylize(image: Image.Image) -> Image.Image | None:
    """Optional stretch pass. Returns None (skip) if diffusers/torch/GPU
    aren't available -- never raises."""
    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore

        if not torch.cuda.is_available():
            return None

        # runwayml/stable-diffusion-v1-5 was taken down from Hugging Face in
        # 2024 (licensing dispute) -- this is the current maintained mirror.
        # Change to a local path (e.g. "checkpoints/sd15") if you'd rather
        # point at weights you've already downloaded.
        model_id = SD15_MODEL_ID
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")
        result = pipe(
            prompt="professional product advertisement photography, clean lighting, commercial",
            image=image.resize((512, 512)),
            strength=0.25,
            guidance_scale=7.0,
        )
        return result.images[0].resize(image.size)
    except Exception as e:
        print(f"[ad_image] img2img stylize skipped: {type(e).__name__}: {e}")
        return None


def generate_ad_image(
    enhanced_image_path: str,
    offer_text: str,
    product: str,
    style: str = "composite",
) -> str:
    with Image.open(enhanced_image_path) as img:
        img = img.convert("RGB")

        if style == "img2img":
            stylized = _try_img2img_stylize(img)
            if stylized is not None:
                img = stylized  # else: silently fall back to plain composite

        ad_image = _composite(img, offer_text, product)

    base = os.path.splitext(os.path.basename(enhanced_image_path))[0]
    out_path = os.path.join(GENERATED_DIR, f"{base}_ad.png")
    ad_image.save(out_path)
    return out_path
