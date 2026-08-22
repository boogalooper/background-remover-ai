from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    try:
        import torch
    except Exception as exc:
        print(f"PyTorch import failed: {exc}")
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA wheel: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        if args.require_cuda:
            print("ERROR: NVIDIA driver was detected, but this PyTorch cannot use CUDA.")
            print("Run install.bat again. The installer must install the CUDA PyTorch build.")
            return 2
        print("NVIDIA CUDA is not available; CPU mode will still work.")
        return 0

    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capability: {torch.cuda.get_device_capability(0)}")
        x = torch.randn((512, 512), device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        print(f"CUDA test OK: {float(y[0,0]):.4f}")
        del x, y
        torch.cuda.empty_cache()
        return 0
    except Exception as exc:
        print(f"CUDA self-test failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
