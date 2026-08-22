from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import load_config, merged_config
from app.core.logging_setup import setup_logging
from app.core.pipeline import BatchPipeline, CancelledError
from app.models.catalog import MODEL_SPECS
from app.paths import configure_runtime_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Background Remover AI — batch local background removal")
    parser.add_argument("sources", nargs="*", help="Files and/or folders")
    parser.add_argument("--output", type=Path, help="Output folder")
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default=None)
    parser.add_argument("--cli", action="store_true", help="Run without GUI")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    configure_runtime_environment()
    setup_logging()
    args = build_parser().parse_args()
    config = load_config()
    overrides: dict = {}
    if args.model:
        overrides.setdefault("model", {})["key"] = args.model
    if args.recursive is not None:
        overrides.setdefault("files", {})["recursive"] = bool(args.recursive)
    if args.overwrite:
        overrides.setdefault("files", {})["overwrite"] = True
    if overrides:
        config = merged_config(config, overrides)

    if args.cli:
        if not args.sources or not args.output:
            print("CLI mode requires sources and --output.", file=sys.stderr)
            return 2
        try:
            pipeline = BatchPipeline(
                config,
                cancel_event=threading.Event(),
                progress=lambda p, m: print(f"\r{p:6.2f}% {m[:100]:100}", end="", flush=True),
                message=lambda m: print(f"\n{m}"),
            )
            stats = pipeline.run([Path(p) for p in args.sources], args.output)
            print("\n")
            print(f"Найдено: {stats.files_found}")
            print(f"Обработано: {stats.files_processed}")
            print(f"Пропущено: {stats.files_skipped}")
            print(f"Ошибок: {stats.files_failed}")
            return 0 if stats.files_failed == 0 else 1
        except CancelledError:
            return 130

    from app.gui.main_window import MainWindow
    MainWindow(
        config,
        initial_sources=args.sources or None,
        initial_output=str(args.output) if args.output else None,
    ).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
