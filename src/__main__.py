"""Единая точка входа: python -m src [1|2|3|all] ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from . import variant1, variant2, variant3
from .common import apply_overrides, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANT_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "1": variant1.run,
    "2": variant2.run,
    "3": variant3.run,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Лабораторная работа №1: шумоподавление/контраст, границы/контуры, "
            "цветовая сегментация."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "variant",
        nargs="?",
        default="all",
        choices=["1", "2", "3", "all"],
        help="номер варианта или all",
    )
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="путь к JSON-конфигу",
    )
    parser.add_argument("--out", default="out", help="папка для результатов")
    parser.add_argument("--data-dir", default="data", help="папка с изображениями")
    parser.add_argument(
        "--data",
        default=None,
        help="путь к своему изображению (переопределяет image из конфига)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="переопределить параметр конфига, например variant1.median_kernel=7",
    )
    return parser


def _resolve_config(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    fallback = PROJECT_ROOT / path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Конфиг не найден: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(_resolve_config(args.config))
    apply_overrides(config, args.overrides)

    out_root = Path(args.out)
    data_dir = Path(args.data_dir)
    variants = list(VARIANT_RUNNERS) if args.variant == "all" else [args.variant]

    for variant in variants:
        section = config.get(f"variant{variant}", {})
        result = VARIANT_RUNNERS[variant](
            section, out_root / f"variant{variant}", data_dir, args.data
        )
        if variant == "1":
            best = result["best_filter_for_example"]
            print(f"[Вариант 1] готово: out/variant1 (лучший фильтр: {best})")
        elif variant == "2":
            counts = {name: item["count"] for name, item in result["methods"].items()}
            print(f"[Вариант 2] готово: out/variant2, объектов: {counts}")
        elif variant == "3":
            classes = {
                item["class"]: item["object_count"] for item in result["classes"]
            }
            print(f"[Вариант 3] готово: out/variant3, объектов: {classes}")

    print(f"Результаты сохранены в: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
