from app.models.catalog import get_model_spec, resolve_batch_size


def test_safe_batch_limits_hr_model():
    spec = get_model_spec("birefnet_hr_matting")
    assert resolve_batch_size(8, spec, "cuda", True) == 1
    assert resolve_batch_size(4, spec, "cuda", False) == 4
    assert resolve_batch_size(4, spec, "cpu", False) == 1


def test_bria_is_marked_gated():
    spec = get_model_spec("bria_rmbg_2")
    assert spec.gated is True
    assert spec.input_size == 1024
