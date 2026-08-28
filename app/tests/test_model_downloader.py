import threading

from app.models.catalog import MODEL_SPECS
from app.models.downloader import download_all_models, download_model


def test_download_all_models_continues_after_one_failure():
    seen = []
    phases = []

    def fake_download(spec):
        seen.append(spec.key)
        if spec.key == "bria_rmbg_2":
            raise RuntimeError("gated")

    result = download_all_models(
        downloader=fake_download,
        progress=lambda i, n, spec, phase: phases.append((i, n, spec.key, phase)),
    )

    assert seen == list(MODEL_SPECS)
    assert result.failed == {"bria_rmbg_2": "gated"}
    assert len(result.ready) == len(MODEL_SPECS) - 1
    assert any(phase == "failed" for _, _, _, phase in phases)
    assert phases[-1][0] == len(MODEL_SPECS)


def test_download_all_models_can_cancel_between_models():
    cancel = threading.Event()
    seen = []

    def fake_download(spec):
        seen.append(spec.key)
        if len(seen) == 1:
            cancel.set()

    result = download_all_models(cancel_event=cancel, downloader=fake_download)

    assert len(seen) == 1
    assert result.cancelled is True
    assert len(result.ready) == 1


def test_download_all_models_retries_transient_failure():
    attempts = {}
    retry_phases = []

    def flaky_download(spec):
        attempts[spec.key] = attempts.get(spec.key, 0) + 1
        if spec.key == "birefnet_portrait" and attempts[spec.key] == 1:
            raise RuntimeError("temporary network reset")

    result = download_all_models(
        downloader=flaky_download,
        retry_delay=0,
        progress=lambda i, n, spec, phase: retry_phases.append((spec.key, phase)),
    )

    assert not result.failed
    assert len(result.ready) == len(MODEL_SPECS)
    assert attempts["birefnet_portrait"] == 2
    assert ("birefnet_portrait", "retry:2:3") in retry_phases


def test_generic_403_can_be_retried_because_cdn_errors_may_be_transient():
    attempts = {}

    def flaky_download(spec):
        attempts[spec.key] = attempts.get(spec.key, 0) + 1
        if spec.key == "bria_rmbg_2" and attempts[spec.key] == 1:
            raise RuntimeError("CAS service error: 403 Forbidden on signed URL")

    result = download_all_models(downloader=flaky_download, retry_delay=0)
    assert not result.failed
    assert attempts["bria_rmbg_2"] == 2


def test_download_model_uses_pinned_revision_and_filtered_files(monkeypatch):
    import huggingface_hub
    from app.models.catalog import get_model_spec

    captured = {}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr("app.models.downloader.get_hf_token", lambda: None)

    spec = get_model_spec("birefnet")
    download_model(spec)

    assert captured["repo_id"] == spec.repo_id
    assert captured["revision"] == spec.revision
    assert captured["allow_patterns"] == ["*.json", "*.py", "model.safetensors"]
