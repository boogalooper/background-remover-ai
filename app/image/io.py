from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


log = logging.getLogger(__name__)

# Only a small, safe subset of TIFF/EXIF metadata is copied into derivative
# TIFF files. Pointer/offset tags and nested IFD structures from the source must
# never be copied to a newly encoded image.
_TIFF_TEXT_TAGS = {
    271,    # Make
    272,    # Model
    315,    # Artist
    316,    # HostComputer
    33432,  # Copyright
}


@dataclass
class LoadedImage:
    path: Path
    image: Image.Image
    icc_profile: bytes | None = None
    exif: bytes | None = None
    dpi: tuple[float, float] | None = None
    source_alpha: Image.Image | None = None


def load_image(path: Path) -> LoadedImage:
    with Image.open(path) as src:
        info = dict(src.info)
        # Apply EXIF orientation before taking either RGB or an existing alpha
        # channel. Otherwise the alpha plane of an oriented RGBA source can be
        # out of sync with the RGB image.
        oriented = ImageOps.exif_transpose(src)
        source_alpha = None
        if "A" in oriented.getbands():
            source_alpha = oriented.getchannel("A").copy()
        image = oriented.convert("RGB").copy()

        exif_bytes = None
        try:
            exif = src.getexif()
            if exif:
                # Orientation was physically applied above.
                exif[274] = 1
                exif_bytes = exif.tobytes()
        except Exception:
            exif_bytes = None

        dpi = info.get("dpi")
        if not (isinstance(dpi, tuple) and len(dpi) >= 2):
            dpi = None

        return LoadedImage(
            path=path,
            image=image,
            icc_profile=info.get("icc_profile"),
            exif=exif_bytes,
            dpi=dpi,
            source_alpha=source_alpha,
        )


def combine_alpha(mask: Image.Image, source_alpha: Image.Image | None) -> Image.Image:
    if source_alpha is None:
        return mask.convert("L")
    from PIL import ImageChops

    if source_alpha.size != mask.size:
        source_alpha = source_alpha.resize(mask.size, Image.Resampling.LANCZOS)
    return ImageChops.multiply(mask.convert("L"), source_alpha.convert("L"))


def _cleanup_tmp(tmp: Path) -> None:
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass


def _validate_pillow_image(path: Path, expected_size: tuple[int, int]) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise OSError(f"Temporary image was not written: {path}")
    with Image.open(path) as check:
        if check.size != expected_size:
            raise OSError(
                f"Saved image has unexpected size {check.size}; expected {expected_size}"
            )
        # verify() checks the container without keeping decoded pixels in RAM.
        check.verify()


def _atomic_save(image: Image.Image, path: Path, *, format_name: str, save_kwargs: dict) -> None:
    """Atomic Pillow save for formats that do not require libtiff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    _cleanup_tmp(tmp)
    try:
        image.save(tmp, format=format_name, **save_kwargs)
        _validate_pillow_image(tmp, image.size)
        os.replace(tmp, path)
    finally:
        _cleanup_tmp(tmp)


def _read_safe_tiff_metadata(exif_bytes: bytes | None) -> dict:
    """Extract metadata that is safe to write into a new TIFF IFD.

    We intentionally do not copy ExifIFD/GPS/Interop pointers, MakerNote, strip
    offsets, JPEG offsets, or other nested/container-specific structures.
    """
    result: dict = {
        "description": None,
        "datetime": None,
        "software": None,
        "extratags": [],
    }
    if not exif_bytes:
        result["extratags"] = [(274, "H", 1, 1, False)]
        return result

    try:
        source = Image.Exif()
        source.load(exif_bytes)

        def _text(tag: int) -> str | None:
            value = source.get(tag)
            if value is None:
                return None
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", "replace")
                except Exception:
                    return None
            value = str(value).replace("\x00", "").strip()
            return value or None

        result["description"] = _text(270)
        result["datetime"] = _text(306)
        result["software"] = _text(305)

        extra = [(274, "H", 1, 1, False)]  # normalized Orientation
        for tag in sorted(_TIFF_TEXT_TAGS):
            value = _text(tag)
            if value:
                extra.append((tag, "s", 0, value, False))
        result["extratags"] = extra
    except Exception:
        log.warning("Could not read safe TIFF metadata; writing image without EXIF fields", exc_info=True)
        result["extratags"] = [(274, "H", 1, 1, False)]
    return result


def _write_tiff_tifffile(
    image: Image.Image,
    loaded: LoadedImage,
    tmp: Path,
    *,
    preserve_metadata: bool,
    compression: str | None,
) -> None:
    """Write RGBA TIFF without Pillow/libtiff.

    `tifffile` writes the TIFF container itself and uses zlib for Deflate. This
    avoids the Windows libtiff crash that can terminate the whole Python process
    while Pillow is encoding a compressed RGBA TIFF.
    """
    import numpy as np
    import tifffile

    rgba = image.convert("RGBA")
    data = np.asarray(rgba, dtype=np.uint8)

    kwargs: dict = {
        "photometric": "rgb",
        "extrasamples": ["unassalpha"],
        "compression": compression,
        "metadata": None,
        # A single compression worker avoids large temporary RAM spikes on
        # 40–60 MP files. CPU prefetching can still happen independently.
        "maxworkers": 1,
        "rowsperstrip": 128,
        "software": False,
    }
    if compression == "deflate":
        kwargs["compressionargs"] = {"level": 6}
        kwargs["predictor"] = True

    if preserve_metadata:
        if loaded.icc_profile:
            kwargs["iccprofile"] = loaded.icc_profile
        if loaded.dpi:
            try:
                xdpi = float(loaded.dpi[0])
                ydpi = float(loaded.dpi[1])
                if xdpi > 0 and ydpi > 0:
                    kwargs["resolution"] = (xdpi, ydpi)
                    kwargs["resolutionunit"] = "INCH"
            except Exception:
                pass

        safe = _read_safe_tiff_metadata(loaded.exif)
        if safe.get("description"):
            kwargs["description"] = safe["description"]
        if safe.get("datetime"):
            kwargs["datetime"] = safe["datetime"]
        if safe.get("software"):
            kwargs["software"] = safe["software"]
        if safe.get("extratags"):
            kwargs["extratags"] = safe["extratags"]

    tifffile.imwrite(tmp, data, **kwargs)


def _validate_tiff(path: Path, expected_size: tuple[int, int]) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise OSError(f"Temporary TIFF was not written: {path}")

    import tifffile

    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise OSError("TIFF contains no image pages")
        page = tif.pages[0]
        expected_shape = (expected_size[1], expected_size[0], 4)
        if tuple(page.shape) != expected_shape:
            raise OSError(
                f"Saved TIFF has unexpected shape {page.shape}; expected {expected_shape}"
            )
        if int(getattr(page, "samplesperpixel", 0) or 0) != 4:
            raise OSError("Saved TIFF does not contain RGBA data")


def _atomic_save_tiff(
    image: Image.Image,
    loaded: LoadedImage,
    path: Path,
    *,
    preserve_metadata: bool,
) -> None:
    """Safely write a TIFF and only publish it after structural validation.

    Primary encoding uses Deflate through tifffile (no libtiff). If that raises a
    normal Python exception, retry uncompressed. A stale/zero-byte `.tmp` from a
    previous interrupted run is always removed before starting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    _cleanup_tmp(tmp)

    log.info("Saving TIFF via tifffile: %s", path)
    try:
        try:
            _write_tiff_tifffile(
                image,
                loaded,
                tmp,
                preserve_metadata=preserve_metadata,
                compression="deflate",
            )
            _validate_tiff(tmp, image.size)
        except Exception:
            log.warning(
                "Deflate TIFF save failed for %s; retrying as uncompressed TIFF",
                path,
                exc_info=True,
            )
            _cleanup_tmp(tmp)
            _write_tiff_tifffile(
                image,
                loaded,
                tmp,
                preserve_metadata=preserve_metadata,
                compression=None,
            )
            _validate_tiff(tmp, image.size)

        os.replace(tmp, path)
        try:
            log.info("TIFF saved: %s (%d bytes)", path, path.stat().st_size)
        except OSError:
            log.info("TIFF saved: %s", path)
    finally:
        _cleanup_tmp(tmp)


def save_cutout(
    loaded: LoadedImage,
    alpha: Image.Image,
    path: Path,
    *,
    format_name: str = "PNG",
    preserve_metadata: bool = True,
) -> None:
    rgba = loaded.image.convert("RGBA")
    rgba.putalpha(combine_alpha(alpha, loaded.source_alpha))
    fmt = format_name.upper()

    if fmt == "TIFF":
        _atomic_save_tiff(rgba, loaded, path, preserve_metadata=preserve_metadata)
        return

    kwargs: dict = {}
    if preserve_metadata:
        if loaded.icc_profile:
            kwargs["icc_profile"] = loaded.icc_profile
        if loaded.exif:
            kwargs["exif"] = loaded.exif
        if loaded.dpi:
            kwargs["dpi"] = loaded.dpi

    if fmt == "PNG":
        kwargs["compress_level"] = 4
    _atomic_save(rgba, path, format_name=fmt, save_kwargs=kwargs)


def save_mask(mask: Image.Image, path: Path) -> None:
    _atomic_save(mask.convert("L"), path, format_name="PNG", save_kwargs={"compress_level": 4})
