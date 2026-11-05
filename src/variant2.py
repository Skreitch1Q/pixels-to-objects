"""Вариант 2. Границы и контуры объектов.

Сравниваются Sobel, Scharr и Canny, маски очищаются морфологией,
контуры фильтруются по площади; для каждого метода выводится число
найденных объектов.
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
    load_image,
    save_bar_chart,
    save_csv,
    save_gray,
    save_json,
    save_montage,
    save_rgb,
)


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _gradient_mask(gray: np.ndarray, method: str, percentile: float) -> np.ndarray:
    if method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    elif method == "scharr":
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    else:
        raise ValueError(f"Неизвестный градиентный метод: {method!r}")

    magnitude = cv2.magnitude(gx, gy)
    threshold = float(np.percentile(magnitude, float(percentile)))
    return (magnitude >= threshold).astype(np.uint8) * 255


def _canny_mask(gray: np.ndarray, low: float, high: float) -> np.ndarray:
    return cv2.Canny(gray, float(low), float(high), L2gradient=True)


def _clean_mask(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    kernel_size = _odd(config["morph_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=int(config["morph_iterations"])
    )

    open_size = int(config.get("morph_open_kernel", 0))
    if open_size >= 3:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(open_size), _odd(open_size))
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    # Удаляем мелкие связные компоненты самой маски.
    min_component_area = int(config.get("min_edge_area", 1))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            filtered[labels == label] = 255
    return filtered


def _contour_metrics(contour: np.ndarray) -> dict[str, float]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": width,
        "bbox_h": height,
        "circularity": float(circularity),
    }


def _annotate(
    gray: np.ndarray,
    contours: list[np.ndarray],
    areas: list[float],
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(canvas, contours, -1, color, 2)
    for index, (contour, area) in enumerate(zip(contours, areas), start=1):
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        cv2.circle(canvas, (cx, cy), 2, (255, 255, 255), -1)
        cv2.putText(
            canvas,
            f"{index}:{area:.0f}",
            (cx + 3, cy - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _plot_area_distribution(
    contours_by_method: dict[str, list[float]], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for method, areas in contours_by_method.items():
        if areas:
            ax.hist(areas, bins=12, alpha=0.55, label=f"{method} (n={len(areas)})")
    ax.set_xlabel("Площадь контура, пиксели")
    ax.set_ylabel("Число объектов")
    ax.set_title("Распределение площадей найденных объектов")
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
    gray = load_image(image_spec, data_dir=data_dir, color=False)
    save_gray(before_dir / "coins.png", gray)

    blur_sigma = float(config.get("pre_blur_sigma", 0.0))
    if blur_sigma > 0:
        work = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    else:
        work = gray.copy()

    methods = ["sobel", "scharr", "canny"]
    raw_masks: dict[str, np.ndarray] = {
        "sobel": _gradient_mask(work, "sobel", config["sobel_percentile"]),
        "scharr": _gradient_mask(work, "scharr", config["scharr_percentile"]),
        "canny": _canny_mask(work, config["canny_low"], config["canny_high"]),
    }
    cleaned_masks = {method: _clean_mask(mask, config) for method, mask in raw_masks.items()}

    for method in methods:
        save_gray(before_dir / f"edges_{method}_raw.png", raw_masks[method])
        save_gray(after_dir / f"edges_{method}_clean.png", cleaned_masks[method])

    min_area = float(config["min_area"])
    max_area = float(config.get("max_area", 0))
    contour_rows: list[dict[str, Any]] = []
    accepted: dict[str, list[np.ndarray]] = {}
    accepted_areas: dict[str, list[float]] = {}
    annotated: dict[str, np.ndarray] = {}
    colors = {
        "sobel": (60, 200, 120),
        "scharr": (70, 150, 255),
        "canny": (255, 120, 70),
    }

    for method in methods:
        contours, _ = cv2.findContours(
            cleaned_masks[method], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        kept: list[np.ndarray] = []
        areas: list[float] = []
        for contour in contours:
            metrics = _contour_metrics(contour)
            area = metrics["area_px"]
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue
            kept.append(contour)
            areas.append(area)
            contour_rows.append(
                {
                    "method": method,
                    "contour_index": len(kept),
                    **metrics,
                }
            )
        accepted[method] = kept
        accepted_areas[method] = areas
        annotated[method] = _annotate(gray, kept, areas, colors[method])
        save_rgb(after_dir / f"contours_{method}.png", annotated[method])

    save_csv(out_dir / "contours.csv", contour_rows)

    save_montage(
        [gray, raw_masks["sobel"], cleaned_masks["sobel"],
         raw_masks["scharr"], cleaned_masks["scharr"],
         raw_masks["canny"], cleaned_masks["canny"]],
        ["Исходное", "Sobel: маска", "Sobel: после морфологии",
         "Scharr: маска", "Scharr: после морфологии",
         "Canny: маска", "Canny: после морфологии"],
        out_dir / "edge_masks.png",
        ncols=3,
        suptitle="Маски границ до и после морфологической очистки",
        vmin=0,
        vmax=255,
    )
    save_montage(
        [gray, annotated["sobel"], annotated["scharr"], annotated["canny"]],
        ["Исходное", "Sobel", "Scharr", "Canny"],
        out_dir / "contours_comparison.png",
        ncols=2,
        suptitle="Найденные объекты после фильтрации по площади",
        vmin=0,
        vmax=255,
    )

    counts = {method: len(accepted[method]) for method in methods}
    mean_areas = {
        method: (float(np.mean(accepted_areas[method])) if accepted_areas[method] else 0.0)
        for method in methods
    }
    save_bar_chart(
        list(counts.keys()),
        list(counts.values()),
        out_dir / "object_counts.png",
        "Число объектов на кадре",
        "Количество контуров",
        color=["#4C78A8", "#F58518", "#E45756"],
    )
    _plot_area_distribution(accepted_areas, out_dir / "area_distribution.png")

    payload = {
        "variant": 2,
        "image": image_spec,
        "image_shape": list(gray.shape),
        "methods": {
            method: {
                "count": counts[method],
                "mean_area_px": mean_areas[method],
                "areas_px": accepted_areas[method],
            }
            for method in methods
        },
        "parameters": {
            "pre_blur_sigma": blur_sigma,
            "sobel_percentile": config["sobel_percentile"],
            "scharr_percentile": config["scharr_percentile"],
            "canny_low": config["canny_low"],
            "canny_high": config["canny_high"],
            "morph_kernel": config["morph_kernel"],
            "morph_iterations": config["morph_iterations"],
            "min_edge_area": config.get("min_edge_area"),
            "min_area": min_area,
            "max_area": max_area,
        },
    }
    save_json(out_dir / "summary.json", payload)
    return payload
