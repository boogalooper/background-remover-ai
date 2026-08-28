from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

GUIDED_EPSILON = 0.002


def remap_mask(mask: Image.Image, black_point: float, white_point: float, gamma: float = 1.0) -> Image.Image:
    black = float(np.clip(black_point, 0.0, 0.999))
    white = float(np.clip(white_point, black + 0.001, 1.0))
    gamma = max(0.1, min(5.0, float(gamma)))
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    arr = np.clip((arr - black) / (white - black), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        arr = np.power(arr, gamma)
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")


def _fallback_pillow_morphology(mask: Image.Image, pixels: int) -> Image.Image:
    pixels = int(pixels)
    if pixels == 0:
        return mask
    remaining = abs(pixels)
    result = mask
    filter_cls = ImageFilter.MaxFilter if pixels > 0 else ImageFilter.MinFilter
    while remaining > 0:
        radius = min(49, remaining)
        result = result.filter(filter_cls(radius * 2 + 1))
        remaining -= radius
    return result


def morphology(mask: Image.Image, pixels: int) -> Image.Image:
    pixels = int(pixels)
    if pixels == 0:
        return mask
    try:
        import cv2
    except Exception:
        return _fallback_pillow_morphology(mask, pixels)

    result = np.asarray(mask.convert("L"), dtype=np.uint8)
    remaining = abs(pixels)
    op = cv2.dilate if pixels > 0 else cv2.erode
    while remaining > 0:
        radius = min(64, remaining)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        result = op(result, kernel, borderType=cv2.BORDER_CONSTANT)
        remaining -= radius
    return Image.fromarray(result, mode="L")


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


def decontaminate_rgba_edges(
    rgba: Image.Image,
    *,
    enabled: bool = True,
    strength: float = 0.5,
    solid_threshold: float = 0.9,
    blur_radius: float = 2.5,
    low_alpha: int = 5,
    high_alpha: int = 250,
) -> Image.Image:
    """Reduce background color spill on semi-transparent edge pixels.

    This is a lightweight local foreground-color estimate.  It intentionally
    affects only the transition zone of the alpha matte and leaves fully opaque
    and fully transparent pixels untouched.
    """
    if not enabled:
        return rgba
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgba
    try:
        import cv2
    except Exception:
        return rgba

    arr = np.asarray(rgba.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    low = float(np.clip(low_alpha / 255.0, 0.0, 1.0))
    high = float(np.clip(high_alpha / 255.0, low + 1e-3, 1.0))
    edge_zone = (alpha > low) & (alpha < high)
    if not np.any(edge_zone):
        return rgba

    solid = np.clip((alpha - solid_threshold) / max(1e-4, 1.0 - solid_threshold), 0.0, 1.0)
    solid = np.maximum(solid, np.power(alpha, 2.0))

    sigma = max(0.5, float(blur_radius))
    denom = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    estimate = np.empty_like(rgb)
    for channel in range(3):
        numer = cv2.GaussianBlur((rgb[..., channel] * solid).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
        estimate[..., channel] = numer / np.maximum(denom, 1e-4)

    zone = np.clip((alpha - low) / (high - low), 0.0, 1.0)
    edge_weight = np.clip(zone * (1.0 - zone) * 4.0, 0.0, 1.0)
    weight = (strength * edge_weight)[..., None]
    rgb = np.where(edge_zone[..., None], rgb * (1.0 - weight) + estimate * weight, rgb)
    out = np.concatenate([np.clip(rgb, 0.0, 1.0), alpha[..., None]], axis=2)
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGBA")


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
            # Epsilon is deliberately an internal algorithm constant.  It used
            # to leak into default.json even though the GUI/presets could not
            # edit or preserve it, which made the public config inconsistent.
            epsilon=GUIDED_EPSILON,
            blend=float(settings.get("guided_blend", 0.35)),
        )
    result = morphology(result, int(settings.get("expand_pixels", 0)))
    result = feather(result, float(settings.get("feather_radius", 0.0)))
    return result
