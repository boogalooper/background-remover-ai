from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path


def _candidate_smi_paths() -> list[Path]:
    paths: list[Path] = []
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    # The helper itself is executed by the 64-bit Python selected by install.bat,
    # so System32 resolves to the real 64-bit system directory even when the
    # installer was launched by a 32-bit parent process (e.g. Total Commander).
    paths.append(windir / "System32" / "nvidia-smi.exe")
    # Sysnative is useful when this helper is ever run from a 32-bit Python.
    paths.append(windir / "Sysnative" / "nvidia-smi.exe")
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        paths.append(Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe")
    return paths


def detect_nvidia() -> tuple[bool, str]:
    if os.name != "nt":
        return False, "Windows NVIDIA detection is only used by install.bat"

    # Most reliable test: the 64-bit NVIDIA display driver exports nvcuda.dll.
    try:
        ctypes.WinDLL("nvcuda.dll")  # type: ignore[attr-defined]
        return True, "NVIDIA CUDA driver detected via nvcuda.dll"
    except OSError:
        pass

    for exe in _candidate_smi_paths():
        try:
            if not exe.is_file():
                continue
            proc = subprocess.run(
                [str(exe), "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode == 0 and "NVIDIA" in proc.stdout.upper():
                first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else str(exe)
                return True, f"NVIDIA detected via nvidia-smi: {first}"
        except Exception:
            continue

    return False, "NVIDIA CUDA driver was not detected"


def main() -> int:
    ok, message = detect_nvidia()
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
