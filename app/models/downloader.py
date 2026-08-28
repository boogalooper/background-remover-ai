from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from app.models.catalog import MODEL_SPECS, ModelSpec
from app.paths import configure_runtime_environment, get_hf_token

log = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int, ModelSpec, str], None]


@dataclass
class ModelDownloadResult:
    ready: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    cancelled: bool = False

def download_model(spec: ModelSpec, *, token: str | None = None) -> None:
    """Populate/complete the Hugging Face cache without loading weights into RAM/GPU."""
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Не найдена библиотека huggingface-hub. Сначала завершите обычную установку через install.bat."
        ) from exc

    kwargs: dict = {
        "repo_id": spec.repo_id,
        # These are the files used by AutoModelForImageSegmentation in this app.
        # Optional ONNX weights, examples and documentation are intentionally skipped.
        "allow_patterns": ["*.json", "*.py", "model.safetensors"],
    }
    if spec.revision:
        kwargs["revision"] = spec.revision
    auth_token = token if token is not None else get_hf_token()
    if auth_token:
        kwargs["token"] = auth_token
    # snapshot_download resumes partial files and reuses a complete local cache.
    snapshot_download(**kwargs)


def _looks_like_permanent_access_error(exc: Exception) -> bool:
    # Do not treat every HTTP 403 as permanent: Hugging Face/Xet/CDN can emit
    # transient 403 responses for signed download URLs.  Gated-repository errors
    # are identified by their exception class or by explicit access wording.
    class_name = exc.__class__.__name__.lower()
    if class_name in {"gatedrepoerror", "repositorynotfounderror"}:
        return True
    text = str(exc).lower()
    markers = (
        "401 unauthorized",
        "gated",
        "cannot access gated repo",
        "gated repo",
        "access to model",
        "access to this resource is restricted",
        "you are not in the authorized list",
        "you must be authenticated",
        "repository not found",
    )
    return any(marker in text for marker in markers)


def download_all_models(
    *,
    cancel_event: threading.Event | None = None,
    progress: ProgressCallback | None = None,
    downloader: Callable[[ModelSpec], None] | None = None,
    max_attempts: int = 3,
    retry_delay: float = 1.0,
) -> ModelDownloadResult:
    """Ensure every configured model is present in cache.

    Transient download/finalization failures are retried automatically.  A model
    is reported as failed only after all attempts have failed.  Permanent access
    errors (for example a gated BRIA repository without permission) are not
    retried pointlessly.
    """
    configure_runtime_environment()
    cancel_event = cancel_event or threading.Event()
    callback = progress or (lambda _i, _n, _spec, _phase: None)
    fetch = downloader or download_model
    specs = list(MODEL_SPECS.values())
    result = ModelDownloadResult()
    attempts_limit = max(1, int(max_attempts))

    for index, spec in enumerate(specs, start=1):
        if cancel_event.is_set():
            result.cancelled = True
            break

        callback(index, len(specs), spec, "start")
        last_exc: Exception | None = None

        for attempt in range(1, attempts_limit + 1):
            if cancel_event.is_set():
                result.cancelled = True
                break
            try:
                fetch(spec)
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Model cache preparation failed for %s (attempt %d/%d): %s",
                    spec.repo_id,
                    attempt,
                    attempts_limit,
                    exc,
                    exc_info=True,
                )
                permanent = _looks_like_permanent_access_error(exc)
                if permanent or attempt >= attempts_limit:
                    break
                callback(index, len(specs), spec, f"retry:{attempt + 1}:{attempts_limit}")
                if retry_delay > 0:
                    # Wait in small chunks so Cancel remains responsive.
                    remaining = float(retry_delay) * attempt
                    while remaining > 0 and not cancel_event.is_set():
                        step = min(0.1, remaining)
                        time.sleep(step)
                        remaining -= step
            else:
                result.ready.append(spec.key)
                callback(index, len(specs), spec, "done")
                last_exc = None
                break

        if result.cancelled:
            break
        if last_exc is not None:
            result.failed[spec.key] = str(last_exc)
            callback(index, len(specs), spec, "failed")

    return result
