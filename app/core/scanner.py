from __future__ import annotations

from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".psd", ".psb"}


def is_supported(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def scan_inputs(
    sources: Iterable[Path],
    *,
    recursive: bool = True,
    exclude_roots: Iterable[Path] = (),
) -> list[Path]:
    excludes = []
    for item in exclude_roots:
        try:
            excludes.append(item.resolve())
        except OSError:
            pass

    def excluded(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for root in excludes:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                pass
        return False

    result: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        source = Path(source).expanduser()
        explicit_file = source.is_file()
        if explicit_file:
            # An explicitly selected file is always an input, even when the user
            # writes results beside it.  exclude_roots is meant to keep recursive
            # folder scans out of their own output directory, not to discard a
            # file the user selected directly.
            candidates = [source]
        elif source.is_dir():
            iterator = source.rglob("*") if recursive else source.glob("*")
            candidates = iterator
        else:
            continue
        for path in candidates:
            if (not explicit_file and excluded(path)) or not is_supported(path):
                continue
            try:
                key = path.resolve()
            except OSError:
                key = path.absolute()
            if key not in seen:
                seen.add(key)
                result.append(path)
    return sorted(result, key=lambda p: str(p).lower())


def common_source_root(sources: list[Path]) -> Path | None:
    dirs = [p if p.is_dir() else p.parent for p in sources]
    if not dirs:
        return None
    if len(dirs) == 1:
        return dirs[0]
    try:
        import os
        return Path(os.path.commonpath([str(d.resolve()) for d in dirs]))
    except Exception:
        return dirs[0]


def suggested_output_dir(sources: Iterable[Path], folder_name: str = "Background Removed") -> Path | None:
    source_list = [Path(s).expanduser() for s in sources]
    root = common_source_root(source_list)
    if root is None:
        return None
    return root / folder_name
