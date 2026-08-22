from pathlib import Path

import numpy as np
from PIL import Image

from app.image.io import LoadedImage, combine_alpha, load_image, save_cutout, save_mask


def test_combine_alpha_multiplies_existing_alpha():
    mask = Image.new("L", (2, 1), 128)
    source = Image.new("L", (2, 1), 128)
    out = combine_alpha(mask, source)
    assert np.asarray(out).tolist() == [[64, 64]]


def test_save_cutout_and_mask_are_rgba_and_l(tmp_path: Path):
    loaded = LoadedImage(path=tmp_path / "source.jpg", image=Image.new("RGB", (4, 3), (10, 20, 30)))
    mask = Image.new("L", (4, 3), 200)
    cutout = tmp_path / "result.png"
    mask_path = tmp_path / "mask.png"
    save_cutout(loaded, mask, cutout)
    save_mask(mask, mask_path)
    with Image.open(cutout) as im:
        assert im.mode == "RGBA"
        assert im.getchannel("A").getextrema() == (200, 200)
    with Image.open(mask_path) as im:
        assert im.mode == "L"
        assert im.getextrema() == (200, 200)


def test_load_image_applies_exif_orientation_to_rgb(tmp_path: Path):
    # 2x3 source becomes 3x2 after Orientation=6.
    image = Image.new("RGB", (2, 3), "white")
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "oriented.jpg"
    image.save(path, exif=exif)
    loaded = load_image(path)
    assert loaded.image.size == (3, 2)
    if loaded.exif:
        recovered = Image.Exif()
        recovered.load(loaded.exif)
        assert recovered.get(274) == 1


def test_tiff_deflate_saves_when_source_exif_contains_nested_exif_ifd(tmp_path: Path):
    """Regression: Pillow/libtiff rejects a copied EXIFIFDOffset as LONG8=-1."""
    source = Image.new("RGB", (12, 8), (100, 120, 140))
    exif = Image.Exif()
    exif[271] = "Test Camera Maker"
    exif[272] = "Test Camera Model"
    exif[274] = 1
    # This nested IFD is what previously produced:
    # Bad LONG8 or IFD8 value 18446744073709551615 for EXIFIFDOffset.
    exif[34665] = {
        33434: (1, 125),
        36867: "2026:08:22 12:34:56",
    }
    source_path = tmp_path / "source.jpg"
    source.save(source_path, exif=exif)

    loaded = load_image(source_path)
    mask = Image.new("L", loaded.image.size, 255)
    output = tmp_path / "result.tif"
    save_cutout(loaded, mask, output, format_name="TIFF", preserve_metadata=True)

    assert output.exists()
    with Image.open(output) as saved:
        assert saved.mode == "RGBA"
        assert saved.getchannel("A").getextrema() == (255, 255)
        saved_exif = saved.getexif()
        assert saved_exif.get(271) == "Test Camera Maker"
        assert saved_exif.get(272) == "Test Camera Model"
        assert saved_exif.get(274) == 1
        # Nested IFDs are intentionally not copied into derivative TIFF files.
        # The important guarantee is that no EXIFIFDOffset pointer is recreated.
        assert saved_exif.get(34665) is None


def test_tiff_writer_does_not_copy_source_tiff_layout_tags(tmp_path: Path):
    image = Image.new("RGBA", (9, 7), (10, 20, 30, 255))
    source_path = tmp_path / "source.tif"
    image.save(source_path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))

    loaded = load_image(source_path)
    mask = Image.new("L", loaded.image.size, 128)
    output = tmp_path / "result.tif"
    save_cutout(loaded, mask, output, format_name="TIFF", preserve_metadata=True)

    with Image.open(output) as saved:
        assert saved.size == (9, 7)
        assert saved.mode == "RGBA"
        assert saved.getchannel("A").getextrema() == (128, 128)


def test_tiff_uses_tifffile_deflate_and_unassociated_alpha(tmp_path: Path):
    import tifffile

    loaded = LoadedImage(
        path=tmp_path / "source.jpg",
        image=Image.new("RGB", (32, 24), (10, 20, 30)),
        dpi=(300.0, 300.0),
    )
    mask = Image.new("L", loaded.image.size, 123)
    output = tmp_path / "safe.tif"
    save_cutout(loaded, mask, output, format_name="TIFF", preserve_metadata=True)

    with tifffile.TiffFile(output) as tif:
        page = tif.pages[0]
        assert tuple(page.shape) == (24, 32, 4)
        assert page.compression.name in {"ADOBE_DEFLATE", "DEFLATE"}
        assert page.extrasamples
        assert "UNASSALPHA" in page.extrasamples[0].name

    with Image.open(output) as saved:
        assert saved.mode == "RGBA"
        assert saved.getchannel("A").getextrema() == (123, 123)


def test_tiff_removes_stale_zero_byte_tmp_before_save(tmp_path: Path):
    loaded = LoadedImage(path=tmp_path / "source.jpg", image=Image.new("RGB", (8, 6), "white"))
    mask = Image.new("L", loaded.image.size, 255)
    output = tmp_path / "result.tif"
    stale = tmp_path / "result.tif.tmp"
    stale.write_bytes(b"")

    save_cutout(loaded, mask, output, format_name="TIFF", preserve_metadata=False)

    assert output.exists()
    assert output.stat().st_size > 0
    assert not stale.exists()


def test_tiff_retries_uncompressed_if_deflate_write_raises(tmp_path: Path, monkeypatch):
    import app.image.io as image_io
    import tifffile

    loaded = LoadedImage(path=tmp_path / "source.jpg", image=Image.new("RGB", (8, 6), "white"))
    mask = Image.new("L", loaded.image.size, 200)
    output = tmp_path / "result.tif"

    real_write = image_io._write_tiff_tifffile
    calls = []

    def flaky(*args, compression=None, **kwargs):
        calls.append(compression)
        if compression == "deflate":
            raise RuntimeError("synthetic compression failure")
        return real_write(*args, compression=compression, **kwargs)

    monkeypatch.setattr(image_io, "_write_tiff_tifffile", flaky)
    save_cutout(loaded, mask, output, format_name="TIFF", preserve_metadata=False)

    assert calls == ["deflate", None]
    with tifffile.TiffFile(output) as tif:
        assert tif.pages[0].compression.name == "NONE"
