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
