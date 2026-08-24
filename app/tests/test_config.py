from app.core.config import merged_config
from app.paths import HF_HOME, configure_runtime_environment, get_hf_token


def test_merged_config_is_deep_and_non_mutating():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = merged_config(base, {"a": {"x": 9}})
    assert out == {"a": {"x": 9, "y": 2}, "b": 3}
    assert base["a"]["x"] == 1


def test_runtime_forces_project_local_huggingface_cache(monkeypatch):
    monkeypatch.setenv("HF_HOME", "X:/external/cache")
    configure_runtime_environment()
    import os
    assert os.environ["HF_HOME"] == str(HF_HOME)


def test_global_hf_token_is_preserved_and_read(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    configure_runtime_environment()
    assert get_hf_token() == "hf_test_secret"


def test_hf_token_legacy_fallback(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy_secret")
    assert get_hf_token() == "legacy_secret"


def test_custom_mask_presets_roundtrip(tmp_path, monkeypatch):
    from app.core import config as config_mod

    target = tmp_path / "custom_mask_presets.json"
    monkeypatch.setattr(config_mod, "CUSTOM_MASK_PRESETS_PATH", target)
    presets = {
        "Волосы": {
            "mask": {
                "black_point": 0.01,
                "white_point": 0.99,
                "gamma": 1.0,
                "expand_pixels": 0,
                "feather_radius": 0.0,
                "guided_refine": True,
                "guided_max_long_edge": 4096,
                "guided_radius": 8,
                "guided_blend": 0.35,
            },
            "cutout": {"decontaminate": True, "decontam_strength": 0.6},
        }
    }
    config_mod.save_custom_mask_presets(presets)
    assert config_mod.load_custom_mask_presets() == presets
