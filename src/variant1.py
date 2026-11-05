"""Вариант 1. Шумоподавление и контраст.

Сравниваются гауссов и медианный фильтры на синтетическом шуме,
а также глобальное выравнивание гистограммы и CLAHE. Строятся
зависимости PSNR/SSIM от уровня шума.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import (
    ensure_dir,
    entropy,
    global_contrast,
    load_image,
    mse,
    psnr,
    save_csv,
    save_gray,
    save_json,
    save_montage,
    save_rgb,
    ssim,
)


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, float(sigma), size=image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_impulse_noise(image: np.ndarray, density: float, rng: np.random.Generator) -> np.ndarray:
    """Соль и перец: доля ``density`` пикселей, поровну чёрных и белых."""
    density = float(np.clip(density, 0.0, 1.0))
    result = image.copy()
    random_map = rng.random(image.shape)
    result[random_map < density / 2.0] = 0
    result[random_map > 1.0 - density / 2.0] = 255
    return result


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def make_filters(config: dict[str, Any]) -> dict[str, Any]:
    gaussian_kernel = _odd_kernel(config["gaussian_kernel"])
    median_kernel = _odd_kernel(config["median_kernel"])
    gaussian_sigma = float(config["gaussian_sigma"])
    return {
        "gaussian": lambda image: cv2.GaussianBlur(
            image, (gaussian_kernel, gaussian_kernel), sigmaX=gaussian_sigma
        ),
        "median": lambda image: cv2.medianBlur(image, median_kernel),
    }


def apply_contrast(image: np.ndarray, method: str, config: dict[str, Any]) -> np.ndarray:
    if method == "none":
        return image.copy()
    if method == "hist_eq":
        return cv2.equalizeHist(image)
    if method == "clahe":
        tile = tuple(int(v) for v in config["clahe_tile_grid_size"])
        clahe = cv2.createCLAHE(
            clipLimit=float(config["clahe_clip_limit"]), tileGridSize=(tile[0], tile[1])
        )
        return clahe.apply(image)
    raise ValueError(f"Неизвестный метод контраста: {method!r}")


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["noise_type"], row["level"], row["method"])
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (noise_type, level, method), items in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        psnr_mean, psnr_std = _mean_std([item["psnr"] for item in items])
        ssim_mean, ssim_std = _mean_std([item["ssim"] for item in items])
        summary.append(
            {
                "noise_type": noise_type,
                "level": level,
                "method": method,
                "trials": len(items),
                "psnr_mean": psnr_mean,
                "psnr_std": psnr_std,
                "ssim_mean": ssim_mean,
                "ssim_std": ssim_std,
            }
        )
    return summary


def _plot_noise_curves(
    summary: list[dict[str, Any]], path: Path, gaussian_label: str, impulse_label: str
) -> None:
    methods = ["gaussian", "median"]
    method_labels = {"gaussian": "Гауссов фильтр", "median": "Медианный фильтр"}
    colors = {"gaussian": "#4C78A8", "median": "#E45756"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cases = [
        ("gaussian", "sigma", axes[0, 0], "PSNR, дБ", gaussian_label),
        ("gaussian", "sigma", axes[0, 1], "SSIM", gaussian_label),
        ("impulse", "density", axes[1, 0], "PSNR, дБ", impulse_label),
        ("impulse", "density", axes[1, 1], "SSIM", impulse_label),
    ]

    for noise_type, level_key, ax, metric_label, x_label in cases:
        for method in methods:
            items = [
                item
                for item in summary
                if item["noise_type"] == noise_type and item["method"] == method
            ]
            x = [item["level"] for item in items]
            if level_key == "density":
                x = [value * 100.0 for value in x]
            if metric_label.startswith("PSNR"):
                y = [item["psnr_mean"] for item in items]
                yerr = [item["psnr_std"] for item in items]
            else:
                y = [item["ssim_mean"] for item in items]
                yerr = [item["ssim_std"] for item in items]
            ax.plot(x, y, marker="o", label=method_labels[method], color=colors[method])
            ax.fill_between(
                x,
                np.asarray(y) - np.asarray(yerr),
                np.asarray(y) + np.asarray(yerr),
                color=colors[method],
                alpha=0.15,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0, 0].set_title("Гауссов шум: PSNR")
    axes[0, 1].set_title("Гауссов шум: SSIM")
    axes[1, 0].set_title("Импульсный шум: PSNR")
    axes[1, 1].set_title("Импульсный шум: SSIM")
    fig.suptitle("Зависимость качества восстановления от уровня шума", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _parameter_sweep(
    image: np.ndarray,
    config: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Чувствительность PSNR/SSIM к параметрам гауссова и медианного фильтров."""
    sweep = config.get("parameter_sweep")
    if not sweep:
        return []

    noise_type = str(sweep.get("noise_type", "gaussian"))
    level = float(sweep["level"])
    trials = int(sweep.get("trials", 2))
    gaussian_kernel = _odd_kernel(sweep.get("gaussian_kernel", config["gaussian_kernel"]))
    gaussian_sigmas = [float(v) for v in sweep["gaussian_sigmas"]]
    median_kernels = [_odd_kernel(v) for v in sweep["median_kernels"]]
    noise_fn = add_gaussian_noise if noise_type == "gaussian" else add_impulse_noise

    rows: list[dict[str, Any]] = []
    for trial in range(trials):
        rng = np.random.default_rng(seed + 777 + trial)
        noisy = noise_fn(image, level, rng)
        for sigma_value in gaussian_sigmas:
            restored = cv2.GaussianBlur(
                noisy, (gaussian_kernel, gaussian_kernel), sigmaX=sigma_value
            )
            rows.append(
                {
                    "filter": "gaussian",
                    "parameter": "sigma",
                    "value": sigma_value,
                    "trial": trial,
                    "psnr": psnr(image, restored),
                    "ssim": ssim(image, restored),
                }
            )
        for kernel_value in median_kernels:
            restored = cv2.medianBlur(noisy, kernel_value)
            rows.append(
                {
                    "filter": "median",
                    "parameter": "kernel",
                    "value": kernel_value,
                    "trial": trial,
                    "psnr": psnr(image, restored),
                    "ssim": ssim(image, restored),
                }
            )

    summary: list[dict[str, Any]] = []
    for filter_name in ("gaussian", "median"):
        values = sorted(
            {row["value"] for row in rows if row["filter"] == filter_name}
        )
        for value in values:
            items = [
                row
                for row in rows
                if row["filter"] == filter_name and row["value"] == value
            ]
            psnr_mean, psnr_std = _mean_std([item["psnr"] for item in items])
            ssim_mean, ssim_std = _mean_std([item["ssim"] for item in items])
            summary.append(
                {
                    "filter": filter_name,
                    "parameter": "sigma" if filter_name == "gaussian" else "kernel",
                    "value": value,
                    "trials": len(items),
                    "psnr_mean": psnr_mean,
                    "psnr_std": psnr_std,
                    "ssim_mean": ssim_mean,
                    "ssim_std": ssim_std,
                }
            )

    save_csv(out_dir / "metrics_parameter_sweep.csv", summary)
    _plot_parameter_sweep(summary, noise_type, level, out_dir / "parameter_sweep.png")
    return summary


def _plot_parameter_sweep(
    summary: list[dict[str, Any]], noise_type: str, level: float, path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    panels = [
        ("gaussian", "PSNR, дБ", axes[0, 0]),
        ("gaussian", "SSIM", axes[0, 1]),
        ("median", "PSNR, дБ", axes[1, 0]),
        ("median", "SSIM", axes[1, 1]),
    ]
    for filter_name, metric_label, ax in panels:
        items = [item for item in summary if item["filter"] == filter_name]
        x = [item["value"] for item in items]
        if metric_label.startswith("PSNR"):
            y = [item["psnr_mean"] for item in items]
            yerr = [item["psnr_std"] for item in items]
        else:
            y = [item["ssim_mean"] for item in items]
            yerr = [item["ssim_std"] for item in items]
        color = "#4C78A8" if filter_name == "gaussian" else "#E45756"
        ax.plot(x, y, marker="o", color=color)
        ax.fill_between(
            x,
            np.asarray(y) - np.asarray(yerr),
            np.asarray(y) + np.asarray(yerr),
            color=color,
            alpha=0.18,
        )
        ax.set_xlabel("σ гауссова фильтра" if filter_name == "gaussian" else "Размер ядра медианного фильтра")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.3)
        ax.set_title(
            ("Гауссов фильтр" if filter_name == "gaussian" else "Медианный фильтр")
            + f": {metric_label}"
        )
    noise_label = "гауссов шум, σ={:g}".format(level) if noise_type == "gaussian" else "импульсный шум, доля={:g}".format(level)
    fig.suptitle(f"Влияние параметров фильтров ({noise_label})", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_histograms(
    original: np.ndarray, enhanced: dict[str, np.ndarray], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for label, image in [("Исходное", original), *enhanced.items()]:
        hist = np.bincount(image.ravel(), minlength=256)
        ax.plot(np.arange(256), hist, label=label, linewidth=1.4)
    ax.set_xlabel("Яркость")
    ax.set_ylabel("Число пикселей")
    ax.set_title("Гистограммы яркости до и после преобразований")
    ax.grid(alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(config: dict[str, Any], out_dir: Path, data_dir: Path, image_override: str | None) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    before_dir = ensure_dir(out_dir / "before")
    after_dir = ensure_dir(out_dir / "after")

    image_spec = image_override or config["image"]
    image = load_image(image_spec, data_dir=data_dir, color=False)
    if image.ndim != 2:
        raise ValueError("Вариант 1 ожидает полутоновое изображение")
    save_gray(before_dir / "source.png", image)

    filters = make_filters(config)
    seed = int(config.get("seed", 42))
    trials = int(config["trials"])

    rows: list[dict[str, Any]] = []
    experiments = (
        ("gaussian", [float(v) for v in config["noise_sigmas"]], add_gaussian_noise),
        ("impulse", [float(v) for v in config["impulse_densities"]], add_impulse_noise),
    )
    for noise_type, levels, noise_fn in experiments:
        for level_index, level in enumerate(levels):
            for trial in range(trials):
                rng = np.random.default_rng(seed + 1000 * trial + level_index)
                noisy = noise_fn(image, level, rng)
                for method, filter_fn in filters.items():
                    restored = filter_fn(noisy)
                    rows.append(
                        {
                            "noise_type": noise_type,
                            "level": level,
                            "method": method,
                            "trial": trial,
                            "psnr": psnr(image, restored),
                            "ssim": ssim(image, restored),
                            "mse": mse(image, restored),
                        }
                    )

    summary = _summarize(rows)
    save_csv(out_dir / "metrics_noise_trials.csv", rows)
    save_csv(out_dir / "metrics_noise_summary.csv", summary)

    gaussian_label = "σ гауссова шума, уровни яркости"
    impulse_label = "Доля импульсного шума, %"
    _plot_noise_curves(summary, out_dir / "noise_psnr_ssim.png", gaussian_label, impulse_label)
    sweep_summary = _parameter_sweep(image, config, seed, out_dir)

    # Репрезентативный пример: тип и уровень шума задаются конфигом.
    example = config["example"]
    example_type = str(example["type"])
    example_level = float(example["level"])
    example_rng = np.random.default_rng(seed + 12345)
    if example_type == "gaussian":
        noisy = add_gaussian_noise(image, example_level, example_rng)
        example_name = f"gaussian_sigma_{example_level:g}"
    elif example_type == "impulse":
        noisy = add_impulse_noise(image, example_level, example_rng)
        example_name = f"impulse_density_{example_level:g}"
    else:
        raise ValueError("example.type должен быть 'gaussian' или 'impulse'")

    # Выбираем лучший фильтр по SSIM в ближайшей точке эксперимента.
    same_type = [item for item in summary if item["noise_type"] == example_type]
    nearest_level = min({item["level"] for item in same_type}, key=lambda v: abs(v - example_level))
    at_level = [item for item in same_type if item["level"] == nearest_level]
    best = max(at_level, key=lambda item: item["ssim_mean"])
    best_method = best["method"]

    filtered = {method: filter_fn(noisy) for method, filter_fn in filters.items()}
    clahe_clean = apply_contrast(image, "clahe", config)
    clahe_restored = apply_contrast(filtered[best_method], "clahe", config)
    hist_eq_clean = apply_contrast(image, "hist_eq", config)

    save_gray(before_dir / f"noisy_{example_name}.png", noisy)
    save_gray(before_dir / "clean_source.png", image)
    for method, result in filtered.items():
        save_gray(after_dir / f"{method}_filter.png", result)
    save_gray(after_dir / "clahe_clean.png", clahe_clean)
    save_gray(after_dir / "hist_eq_clean.png", hist_eq_clean)
    save_gray(after_dir / f"{best_method}_filter_clahe.png", clahe_restored)

    save_montage(
        [
            image,
            noisy,
            filtered["gaussian"],
            filtered["median"],
            hist_eq_clean,
            clahe_clean,
        ],
        [
            "Исходное",
            f"Зашумлённое ({example_name})",
            "Гауссов фильтр",
            "Медианный фильтр",
            "Выравнивание гистограммы",
            "CLAHE",
        ],
        out_dir / "filtering_comparison.png",
        ncols=3,
        cmap="gray",
        suptitle="Шумоподавление и повышение контраста",
        vmin=0,
        vmax=255,
    )

    save_montage(
        [image, noisy, filtered[best_method], clahe_restored],
        [
            "Исходное",
            f"Зашумлённое ({example_name})",
            f"Лучший фильтр ({best_method})",
            "Фильтр + CLAHE",
        ],
        out_dir / "pipeline_comparison.png",
        ncols=4,
        cmap="gray",
        suptitle="Совместный пайплайн: остаточный шум усиливается CLAHE",
        vmin=0,
        vmax=255,
    )

    # Сравнение глобального выравнивания и CLAHE по объективным метрикам.
    contrast_rows: list[dict[str, Any]] = []
    for method in ("hist_eq", "clahe"):
        enhanced = apply_contrast(image, method, config)
        row = {
            "method": method,
            "psnr_vs_original": psnr(image, enhanced),
            "ssim_vs_original": ssim(image, enhanced),
            "mean": float(np.mean(enhanced)),
            "std_contrast": global_contrast(enhanced),
            "entropy": entropy(enhanced),
        }
        contrast_rows.append(row)

    clean_row = {
        "method": "none",
        "psnr_vs_original": None,
        "ssim_vs_original": 1.0,
        "mean": float(np.mean(image)),
        "std_contrast": global_contrast(image),
        "entropy": entropy(image),
    }
    contrast_rows.insert(0, clean_row)
    save_csv(out_dir / "metrics_contrast.csv", contrast_rows)

    save_montage(
        [image, hist_eq_clean, clahe_clean],
        ["Исходное", "Выравнивание гистограммы", "CLAHE"],
        out_dir / "contrast_comparison.png",
        ncols=3,
        cmap="gray",
        suptitle="Сравнение методов повышения контраста",
        vmin=0,
        vmax=255,
    )
    _plot_histograms(
        image,
        {"Выравнивание гистограммы": hist_eq_clean, "CLAHE": clahe_clean},
        out_dir / "contrast_histograms.png",
    )

    pipeline_rows = [
        {
            "stage": "noisy",
            "psnr_vs_original": psnr(image, noisy),
            "ssim_vs_original": ssim(image, noisy),
        },
        {
            "stage": f"{best_method}_filter",
            "psnr_vs_original": psnr(image, filtered[best_method]),
            "ssim_vs_original": ssim(image, filtered[best_method]),
        },
        {
            "stage": f"{best_method}_filter_clahe",
            "psnr_vs_original": psnr(image, clahe_restored),
            "ssim_vs_original": ssim(image, clahe_restored),
        },
    ]
    save_csv(out_dir / "metrics_pipeline.csv", pipeline_rows)

    payload = {
        "variant": 1,
        "image": image_spec,
        "image_shape": list(image.shape),
        "noise_summary": summary,
        "parameter_sweep": sweep_summary,
        "contrast": contrast_rows,
        "pipeline": pipeline_rows,
        "best_filter_for_example": best_method,
        "example": {"type": example_type, "level": example_level},
    }
    save_json(out_dir / "summary.json", payload)
    return payload
