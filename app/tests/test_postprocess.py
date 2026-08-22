import numpy as np
from PIL import Image

from app.image.postprocess import feather, morphology, process_mask, remap_mask


def test_remap_mask_clips_black_and_white_points():
    mask = Image.fromarray(np.array([[0, 64, 128, 192, 255]], dtype=np.uint8))
    out = remap_mask(mask, 0.25, 0.75, 1.0)
    values = np.asarray(out)[0].tolist()
    assert values[0] == 0
    assert values[1] <= 1
    assert 120 <= values[2] <= 136
    assert values[3] == 255
    assert values[4] == 255


def test_morphology_expand_and_shrink():
    arr = np.zeros((7, 7), dtype=np.uint8)
    arr[3, 3] = 255
    mask = Image.fromarray(arr)
    expanded = morphology(mask, 1)
    assert np.asarray(expanded).sum() == 9 * 255
    shrunk = morphology(expanded, -1)
    assert np.asarray(shrunk)[3, 3] == 255


def test_process_mask_noop_profile_preserves_size():
    mask = Image.new("L", (31, 17), 128)
    rgb = Image.new("RGB", (31, 17), "gray")
    out = process_mask(mask, rgb, {
        "black_point": 0.0, "white_point": 1.0, "gamma": 1.0,
        "guided_refine": False, "expand_pixels": 0, "feather_radius": 0.0,
    })
    assert out.size == mask.size
    assert out.getextrema() == (128, 128)


def test_large_edge_shift_has_obvious_effect_on_full_resolution_mask():
    arr = np.zeros((200, 200), dtype=np.uint8)
    arr[50:150, 50:150] = 255
    mask = Image.fromarray(arr)
    expanded = morphology(mask, 30)
    shrunk = morphology(mask, -30)
    # 30 output pixels must visibly move each edge, not act in model-resolution units.
    assert np.asarray(expanded)[25, 100] == 255
    assert np.asarray(shrunk)[85, 100] == 255
    assert np.asarray(shrunk)[60, 100] == 0


def test_edge_shift_is_not_silently_capped_at_49_pixels():
    arr = np.zeros((400, 400), dtype=np.uint8)
    arr[150:250, 150:250] = 255
    mask = Image.fromarray(arr)
    expanded = morphology(mask, 80)
    # The original left edge is x=150. +80 must reach x=70.
    assert np.asarray(expanded)[200, 75] == 255
    assert np.asarray(expanded)[200, 60] == 0
