from pathlib import Path

from PIL import Image

from app.core.pipeline import BatchPipeline, OutputPaths


class FakeBackend:
    def __init__(self, spec, **kwargs):
        self.spec = spec
        self.device = "cpu"
        self.batch_size = 2
        self.loaded = False
        self.closed = False

    def load(self):
        self.loaded = True

    def predict(self, images):
        return [Image.new("L", image.size, 255) for image in images]

    def close(self):
        self.closed = True


def make_config(mode="both"):
    return {
        "model": {"key": "bria_rmbg_2"},
        "files": {
            "recursive": True,
            "preserve_structure": True,
            "overwrite": False,
            "output_mode": mode,
            "cutout_format": "png",
            "cutout_suffix": "_cutout",
            "mask_suffix": "_mask",
            "preserve_metadata": False,
        },
        "mask": {
            "black_point": 0.0,
            "white_point": 1.0,
            "gamma": 1.0,
            "guided_refine": False,
            "expand_pixels": 0,
            "feather_radius": 0.0,
        },
        "performance": {
            "device": "cpu",
            "fp16": False,
            "safe_gpu_memory": True,
            "gpu_batch_size": 2,
            "prefetch_workers": 2,
            "prefetch_buffer": 3,
        },
    }


def test_missing_outputs_does_not_overwrite_existing_half(tmp_path: Path):
    cutout = tmp_path / "a_cutout.png"
    mask = tmp_path / "a_mask.png"
    cutout.write_bytes(b"existing")
    p = BatchPipeline(make_config(), backend_factory=FakeBackend)
    filtered = p._missing_outputs(OutputPaths(cutout=cutout, mask=mask), overwrite=False)
    assert filtered.cutout is None
    assert filtered.mask == mask


def test_pipeline_batch_writes_cutout_and_mask_and_monotonic_progress(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    sub = src / "class"
    sub.mkdir(parents=True)
    for idx in range(3):
        Image.new("RGB", (20 + idx, 15), (10, 20, 30)).save(sub / f"img{idx}.jpg")
    progress = []
    pipeline = BatchPipeline(
        make_config("both"), backend_factory=FakeBackend,
        progress=lambda p, m: progress.append((p, m)),
    )
    stats = pipeline.run([src], out)
    assert stats.files_found == 3
    assert stats.files_processed == 3
    assert stats.files_failed == 0
    assert stats.cutouts_written == 3
    assert stats.masks_written == 3
    assert (out / "class" / "img0_cutout.png").exists()
    assert (out / "class" / "img0_mask.png").exists()
    values = [p for p, _ in progress]
    assert values == sorted(values)
    assert values[-1] == 100.0


def test_pipeline_rerun_creates_only_missing_output(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir(); out.mkdir()
    Image.new("RGB", (10, 10), "red").save(src / "a.jpg")
    existing = out / "a_cutout.png"
    existing.write_bytes(b"keep-me")
    original = existing.read_bytes()
    pipeline = BatchPipeline(make_config("both"), backend_factory=FakeBackend)
    stats = pipeline.run([src], out)
    assert existing.read_bytes() == original
    assert (out / "a_mask.png").exists()
    assert stats.cutouts_written == 0
    assert stats.masks_written == 1

class OOMBackend(FakeBackend):
    def predict(self, images):
        if len(images) > 1:
            raise RuntimeError("Недостаточно VRAM для выбранного GPU-пакета")
        return [Image.new("L", images[0].size, 255)]


def test_pipeline_automatically_splits_batch_after_oom(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    for idx in range(3):
        Image.new("RGB", (12, 9), "blue").save(src / f"p{idx}.jpg")
    messages = []
    pipeline = BatchPipeline(make_config("cutout"), backend_factory=OOMBackend, message=messages.append)
    stats = pipeline.run([src], out)
    assert stats.files_processed == 3
    assert stats.files_failed == 0
    assert any("автоматически повторяю" in message for message in messages)


def test_pipeline_existing_outputs_skip_before_backend_load_and_warn(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir(); out.mkdir()
    Image.new("RGB", (10, 10), "red").save(src / "a.jpg")
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(out / "a_cutout.png")

    calls = {"created": 0}
    messages = []

    def factory(*args, **kwargs):
        calls["created"] += 1
        return FakeBackend(*args, **kwargs)

    pipeline = BatchPipeline(make_config("cutout"), backend_factory=factory, message=messages.append)
    stats = pipeline.run([src], out)
    assert stats.files_processed == 0
    assert stats.files_skipped == 1
    assert calls["created"] == 0
    assert any("НЕ применяются" in message for message in messages)


class SquareMaskBackend(FakeBackend):
    def predict(self, images):
        masks = []
        for image in images:
            w, h = image.size
            mask = Image.new("L", (w, h), 0)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.rectangle((w // 4, h // 4, 3 * w // 4 - 1, 3 * h // 4 - 1), fill=255)
            masks.append(mask)
        return masks


def test_pipeline_applies_edge_shift_to_saved_alpha(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (100, 100), "red").save(src / "a.jpg")

    base_cfg = make_config("cutout")
    base_cfg["files"]["overwrite"] = True

    plus_cfg = make_config("cutout")
    plus_cfg["files"]["overwrite"] = True
    plus_cfg["mask"]["expand_pixels"] = 20
    minus_cfg = make_config("cutout")
    minus_cfg["files"]["overwrite"] = True
    minus_cfg["mask"]["expand_pixels"] = -20

    out_plus = tmp_path / "plus"
    out_minus = tmp_path / "minus"
    BatchPipeline(plus_cfg, backend_factory=SquareMaskBackend).run([src], out_plus)
    BatchPipeline(minus_cfg, backend_factory=SquareMaskBackend).run([src], out_minus)

    with Image.open(out_plus / "a_cutout.png") as plus, Image.open(out_minus / "a_cutout.png") as minus:
        a_plus = plus.getchannel("A")
        a_minus = minus.getchannel("A")
        assert a_plus.getpixel((2, 50)) == 0
        assert a_plus.getpixel((10, 50)) == 255  # original edge=25, expanded by 20 -> about 5
        assert a_minus.getpixel((30, 50)) == 0
        assert a_minus.getpixel((48, 50)) == 255  # original edge=25, shrunk by 20 -> about 45
