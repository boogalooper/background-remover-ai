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


def test_model_recommended_overrides_only_contains_quality_settings():
    from app.models.catalog import MODEL_SPECS, model_recommended_overrides

    for key in MODEL_SPECS:
        overrides = model_recommended_overrides(key)
        assert set(overrides) == {"mask", "cutout"}
        assert "gpu_batch_size" not in overrides.get("performance", {})


def test_all_models_have_compact_gui_hint():
    from app.models.catalog import MODEL_SPECS
    assert MODEL_SPECS
    for spec in MODEL_SPECS.values():
        assert spec.compact_hint.strip()
        assert len(spec.compact_hint) <= 260
