"""Общие утилиты: конфиг, загрузка данных, метрики и сохранение иллюстраций."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")  # работа без графического дисплея
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

BUILTIN_IMAGES = ("camera", "coins", "astronaut", "page", "chelsea", "coffee")


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    """Читает JSON-конфиг и возвращает словарь."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def set_config_value(config: dict[str, Any], dotted_key: str, value: str) -> None:
    """Переопределяет параметр конфига по ключу вида ``variant1.gaussian_sigma``."""
    keys = dotted_key.split(".")
    node: dict[str, Any] = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = _parse_scalar(value)


def apply_overrides(config: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Ожидается KEY=VALUE, получено: {item!r}")
        key, value = item.split("=", 1)
        set_config_value(config, key.strip(), value.strip())
    return config


# ---------------------------------------------------------------------------
# Загрузка изображений
# ---------------------------------------------------------------------------
def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _prepare_builtin(name: str, color: bool) -> np.ndarray:
    if name not in BUILTIN_IMAGES:
        raise ValueError(
            f"Неизвестное встроенное изображение {name!r}. "
            f"Доступны: {', '.join(BUILTIN_IMAGES)}"
        )
    image = getattr(data, name)()
    if color and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if not color and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(image)


def load_image(spec: str | Path, data_dir: str | Path = "data", color: bool = False) -> np.ndarray:
    """Загружает изображение из файла или из встроенного набора scikit-image.

    Найденные встроенные изображения кэшируются в ``data_dir/skimage``,
    поэтому после первого запуска работа не требует сети.
    """
    data_dir = Path(data_dir)
    spec = str(spec)
    candidate = Path(spec)

    if candidate.exists():
        if color:
            bgr = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"Не удалось прочитать изображение: {candidate}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise OSError(f"Не удалось прочитать изображение: {candidate}")
        return gray

    cache_path = data_dir / "skimage" / f"{spec}.png"
    if cache_path.exists():
        return load_image(cache_path, data_dir=data_dir, color=color)

    image = _prepare_builtin(spec, color=color)
    try:
        if color:
            _save_rgb(cache_path, image)
        else:
            _save_gray(cache_path, image)
    except OSError:
        # Кэш может быть недоступен только для чтения — это не критично.
        pass
    return image


# ---------------------------------------------------------------------------
# Ввод-вывод и метрики
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def save_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(reference, test, data_range=255))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    if reference.ndim == 3:
        return float(
            structural_similarity(
                reference, test, data_range=255, channel_axis=-1
            )
        )
    return float(structural_similarity(reference, test, data_range=255))


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    diff = reference.astype(np.float64) - test.astype(np.float64)
    return float(np.mean(diff * diff))


def global_contrast(image: np.ndarray) -> float:
    return float(np.std(image.astype(np.float64)))


def entropy(image: np.ndarray) -> float:
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def save_montage(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    path: str | Path,
    ncols: int = 3,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    suptitle: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Сохраняет сетку изображений с подписями (поддерживает кириллицу)."""
    if len(images) != len(titles):
        raise ValueError("Число изображений и подписей должно совпадать")
    ncols = max(1, min(ncols, len(images)))
    nrows = int(np.ceil(len(images) / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, image, title in zip(axes_arr, images, titles):
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap or "gray", vmin=vmin, vmax=vmax)
        else:
            ax.imshow(image, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes_arr[len(images):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    path: str | Path,
    title: str,
    ylabel: str,
    color: str | Sequence[str] = "#4C78A8",
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
