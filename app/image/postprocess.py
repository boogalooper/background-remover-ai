from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter


def remap_mask(mask: Image.Image, black_point: float, white_point: float, gamma: float = 1.0) -> Image.Image:
    black = float(np.clip(black_point, 0.0, 0.999))
    white = float(np.clip(white_point, black + 0.001, 1.0))
    gamma = max(0.1, min(5.0, float(gamma)))
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    arr = np.clip((arr - black) / (white - black), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        arr = np.power(arr, gamma)
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")


def morphology(mask: Image.Image, pixels: int) -> Image.Image:
    pixels = int(pixels)
    if pixels == 0:
        return mask

    # Pillow's rank filters become impractical with very large kernels. Apply
    # several <=99px kernels instead. For a square structuring element,
    # repeated dilation/erosion adds the requested radii, so +100 really means
    # roughly 100 pixels on the final full-resolution matte rather than being
    # silently capped at 49 pixels.
    remaining = abs(pixels)
    result = mask
    filter_cls = ImageFilter.MaxFilter if pixels > 0 else ImageFilter.MinFilter
    while remaining > 0:
        radius = min(49, remaining)
        result = result.filter(filter_cls(radius * 2 + 1))
        remaining -= radius
    return result


def feather(mask: Image.Image, radius: float) -> Image.Image:
    radius = max(0.0, float(radius))
    if radius <= 0.0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def guided_refine_mask(
    rgb: Image.Image,
    mask: Image.Image,
    *,
    max_long_edge: int = 4096,
    radius: int = 8,
    epsilon: float = 0.002,
    blend: float = 0.35,
) -> Image.Image:
    """Conservative guided-filter refinement aligned to high-resolution luminance.

    It is intentionally optional.  The model remains the primary source of the
    matte; this only nudges the transition zone toward edges in the original.
    """
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0:
        return mask
    try:
        import cv2
    except Exception:
        return mask

    w, h = rgb.size
    scale = min(1.0, float(max_long_edge) / max(w, h))
    rw = max(32, int(round(w * scale)))
    rh = max(32, int(round(h * scale)))
    guide_img = rgb.resize((rw, rh), Image.Resampling.BILINEAR).convert("L")
    mask_img = mask.resize((rw, rh), Image.Resampling.BILINEAR).convert("L")
    guide = np.asarray(guide_img, dtype=np.float32) / 255.0
    p = np.asarray(mask_img, dtype=np.float32) / 255.0
    r = max(1, int(radius))
    k = (2 * r + 1, 2 * r + 1)

    mean_i = cv2.boxFilter(guide, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(p, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide * guide, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide * p, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + max(1e-6, float(epsilon)))
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    refined = mean_a * guide + mean_b
    mixed = np.clip(p * (1.0 - blend) + refined * blend, 0.0, 1.0)
    out = Image.fromarray(np.clip(mixed * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
    return out.resize((w, h), Image.Resampling.LANCZOS)


def process_mask(mask: Image.Image, rgb: Image.Image, settings: dict) -> Image.Image:
    result = remap_mask(
        mask,
        float(settings.get("black_point", 0.0)),
        float(settings.get("white_point", 1.0)),
        float(settings.get("gamma", 1.0)),
    )
    if bool(settings.get("guided_refine", False)):
        result = guided_refine_mask(
            rgb,
            result,
            max_long_edge=int(settings.get("guided_max_long_edge", 4096)),
            radius=int(settings.get("guided_radius", 8)),
            epsilon=float(settings.get("guided_epsilon", 0.002)),
            blend=float(settings.get("guided_blend", 0.35)),
        )
    result = morphology(result, int(settings.get("expand_pixels", 0)))
    result = feather(result, float(settings.get("feather_radius", 0.0)))
    return result
