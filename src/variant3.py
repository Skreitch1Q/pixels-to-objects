"""Вариант 3. Сегментация по цвету в HSV.

Пороги для 2-3 классов объектов задаются в конфиге, маски очищаются
морфологией, объекты и их площади считаются через regionprops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops

from .common import (
    ensure_dir,
    load_image,
    save_csv,
    save_gray,
    save_json,
    save_montage,
    save_rgb,
)


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _class_mask(hsv: np.ndarray, ranges: list[list[list[int]]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        low = np.asarray(lower, dtype=np.uint8)
        high = np.asarray(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    return mask


def _clean_mask(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    kernel_size = _odd(config.get("morph_kernel", 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    iterations = int(config.get("morph_iterations", 1))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Удаляем мелкие связные компоненты (совместимо с разными версиями OpenCV).
    min_area = int(config.get("morph_min_area", 25))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == component] = 255
    return filtered


def _region_objects(
    mask: np.ndarray, min_area: float
) -> tuple[np.ndarray, list[Any]]:
    labeled = label(mask, connectivity=2)
    props = [prop for prop in regionprops(labeled) if prop.area >= min_area]
    filtered = np.zeros_like(mask)
    for prop in props:
        filtered[labeled == prop.label] = 255
    return filtered, props


def _annotate(
    image: np.ndarray,
    filtered_masks: dict[str, np.ndarray],
    objects: dict[str, list[Any]],
    class_config: dict[str, dict[str, Any]],
    alpha: float,
) -> np.ndarray:
    color_layer = image.copy()
    for name, mask in filtered_masks.items():
        color = tuple(int(v) for v in class_config[name]["color"])
        color_layer[mask > 0] = color

    canvas = cv2.addWeighted(image, 1.0 - alpha, color_layer, alpha, 0)
    for name, mask in filtered_masks.items():
        color = tuple(int(v) for v in class_config[name]["color"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
        for index, prop in enumerate(objects[name], start=1):
            cy, cx = (int(v) for v in prop.centroid)
            cv2.putText(
                canvas,
                f"{name[:3]}{index}",
                (cx + 2, cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def _plot_class_stats(
    names: list[str], titles: list[str], counts: list[int], fractions: list[float], path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = ["#F58518", "#E45756", "#4C78A8", "#72B7B2", "#54A24B"]
    axes[0].bar(titles, counts, color=colors[: len(titles)])
    axes[0].set_title("Количество объектов")
    axes[0].set_ylabel("шт.")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(axis="y", alpha=0.3)
    for index, value in enumerate(counts):
        axes[0].text(index, value, str(value), ha="center", va="bottom")

    axes[1].bar(titles, fractions, color=colors[: len(titles)])
    axes[1].set_title("Доля площади кадра")
    axes[1].set_ylabel("%")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.3)
    for index, value in enumerate(fractions):
        axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Результаты сегментации по цвету")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(config: dict[str, Any], out_dir: Path, data_dir: Path, image_override: str | None) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    before_dir = ensure_dir(out_dir / "before")
    after_dir = ensure_dir(out_dir / "after")

    image_spec = image_override or config["image"]
    image = load_image(image_spec, data_dir=data_dir, color=True)
    save_rgb(before_dir / "source.png", image)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    classes = list(config["classes"])
    if not 2 <= len(classes) <= 3:
        raise ValueError("Для варианта 3 ожидается 2 или 3 класса объектов")

    class_config = {str(item["name"]): item for item in classes}
    raw_masks: dict[str, np.ndarray] = {}
    cleaned_masks: dict[str, np.ndarray] = {}
    for item in classes:
        name = str(item["name"])
        raw = _class_mask(hsv, item["hsv_ranges"])
        raw_masks[name] = raw
        cleaned_masks[name] = _clean_mask(raw, config)
        save_gray(before_dir / f"mask_{name}_raw.png", raw)
        save_gray(after_dir / f"mask_{name}_clean.png", cleaned_masks[name])

    # Классы делаем взаимоисключающими: при пересечении приоритет у класса,
    # который указан в конфиге раньше.
    occupied = np.zeros(image.shape[:2], dtype=np.uint8)
    filtered_masks: dict[str, np.ndarray] = {}
    objects: dict[str, list[Any]] = {}
    rows: list[dict[str, Any]] = []
    frame_area = float(image.shape[0] * image.shape[1])

    for item in classes:
        name = str(item["name"])
        mask = cv2.bitwise_and(cleaned_masks[name], cv2.bitwise_not(occupied))
        min_area = float(item.get("min_area", config.get("min_area", 1)))
        filtered, props = _region_objects(mask, min_area)
        filtered_masks[name] = filtered
        objects[name] = props
        occupied = cv2.bitwise_or(occupied, filtered)
        for index, prop in enumerate(props, start=1):
            min_row, min_col, max_row, max_col = prop.bbox
            rows.append(
                {
                    "class": name,
                    "object_index": index,
                    "area_px": int(prop.area),
                    "area_pct": 100.0 * float(prop.area) / frame_area,
                    "centroid_y": float(prop.centroid[0]),
                    "centroid_x": float(prop.centroid[1]),
                    "bbox_top": int(min_row),
                    "bbox_left": int(min_col),
                    "bbox_bottom": int(max_row),
                    "bbox_right": int(max_col),
                    "equivalent_diameter": float(prop.equivalent_diameter_area),
                    "perimeter": float(prop.perimeter),
                    "solidity": float(prop.solidity),
                }
            )

    save_csv(out_dir / "regions.csv", rows)

    overlay = _annotate(
        image,
        filtered_masks,
        objects,
        class_config,
        float(config.get("overlay_alpha", 0.4)),
    )
    save_rgb(after_dir / "segmentation_overlay.png", overlay)

    objects_montage: list[np.ndarray] = [image]
    titles: list[str] = ["Исходное"]
    for item in classes:
        name = str(item["name"])
        objects_montage.append(raw_masks[name])
        titles.append(f"{item['title']}: исходная маска")
    for item in classes:
        name = str(item["name"])
        objects_montage.append(filtered_masks[name])
        titles.append(f"{item['title']}: после очистки")
    objects_montage.append(overlay)
    titles.append("Итоговая сегментация")
    save_montage(
        objects_montage,
        titles,
        out_dir / "segmentation_steps.png",
        ncols=3,
        suptitle="Сегментация по цвету: маски и объекты",
        vmin=0,
        vmax=255,
    )

    class_summary: list[dict[str, Any]] = []
    names: list[str] = []
    display_titles: list[str] = []
    counts: list[int] = []
    fractions: list[float] = []
    for item in classes:
        name = str(item["name"])
        props = objects[name]
        total_area = int(sum(prop.area for prop in props))
        fraction = 100.0 * total_area / frame_area
        names.append(name)
        display_titles.append(str(item["title"]))
        counts.append(len(props))
        fractions.append(fraction)
        class_summary.append(
            {
                "class": name,
                "title": item["title"],
                "object_count": len(props),
                "total_area_px": total_area,
                "area_pct": fraction,
                "mean_object_area_px": (total_area / len(props) if props else 0.0),
            }
        )

    _plot_class_stats(names, display_titles, counts, fractions, out_dir / "class_stats.png")

    segmented_pixels = int(np.count_nonzero(occupied))
    payload = {
        "variant": 3,
        "image": image_spec,
        "image_shape": list(image.shape),
        "classes": class_summary,
        "total_segmented_pct": 100.0 * segmented_pixels / frame_area,
        "parameters": {
            "morph_kernel": config.get("morph_kernel"),
            "morph_iterations": config.get("morph_iterations"),
            "min_area": config.get("min_area"),
            "overlay_alpha": config.get("overlay_alpha"),
        },
    }
    save_json(out_dir / "summary.json", payload)
    return payload
