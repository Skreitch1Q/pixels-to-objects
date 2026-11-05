# Лабораторная работа №1 — обработка изображений

Реализованы все три варианта задания единым CLI. В работе используются
пользовательские изображения из `data/`:

| Вариант | Файл | Размер (высота × ширина) | Содержание |
|---|---|---|---|
| 1 | `data/images.jpg` | 446 × 448 | корабль и акула, полутоновая обработка |
| 2 | `data/images2.jpg` | 621 × 494 | кит и корабль, выделение границ |
| 3 | `data/images3.jpg` | 459 × 435 | корабль в пасти акулы, цветовая сегментация |

Все параметры вынесены в JSON-конфиг `configs/default.json` и могут
переопределяться аргументами командной строки. Результаты (изображения
«до/после», маски, графики и CSV с метриками) сохраняются в `out/`.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# все варианты
python -m src all

# отдельный вариант
python -m src 1
python -m src 2
python -m src 3
```

Запуск без активации окружения:

```bash
.venv/bin/python -m src all --config configs/default.json --out out
```

## Требования

- Python 3.10+ (проверено на 3.14);
- NumPy, OpenCV, Pillow, scikit-image, matplotlib — см. `requirements.txt`.

Изображения по умолчанию берутся из `data/` по именам, указанным в конфиге.
Дополнительно `src/common.py` поддерживает встроенные изображения
`skimage.data` (`camera`, `coins`, `astronaut` и др.): если имя не найдено
среди файлов, оно загружается из scikit-image и кэшируется в
`data/skimage/`.

## Варианты

### Вариант 1 — шумоподавление и контраст

Используется `data/images.jpg` (446 × 448), переведённое в полутоновой вид.

- гауссов и медианный фильтры;
- глобальное выравнивание гистограммы и CLAHE;
- синтетический гауссов и импульсный («соль и перец») шум;
- PSNR/SSIM для каждого уровня шума, усреднение по нескольким реализациям,
  графики зависимости с доверительной полосой;
- сравнение контрастных преобразований по PSNR/SSIM, среднему, СКО и энтропии.

```bash
python -m src 1
python -m src 1 --set variant1.example.level=35 \
                --set variant1.clahe_clip_limit=3.0 \
                --set variant1.clahe_tile_grid_size='[4,4]'
```

### Вариант 2 — границы и контуры

Используется `data/images2.jpg` (621 × 494): кит слева и корабль справа,
крупная подпись внизу.

- Sobel, Scharr (порог по перцентилю модуля градиента) и Canny;
- морфологическое закрытие и удаление мелких связных компонент маски;
- поиск внешних контуров, фильтрация по площади;
- количество объектов на кадре, площади, периметры, компактность;
- разметка найденных объектов и график распределения площадей.

```bash
python -m src 2
python -m src 2 --set variant2.min_area=25000 \
                --set variant2.canny_low=60 \
                --set variant2.canny_high=180
```

### Вариант 3 — сегментация по цвету

Используется `data/images3.jpg` (459 × 435). Заданы три цветовых класса:

- зелёная палуба корабля;
- розовая пасть акулы;
- синяя морская вода.

```bash
python -m src 3
python -m src 3 --set variant3.overlay_alpha=0.55
```

- перевод RGB → HSV;
- пороги классов задаются в `variant3.classes[].hsv_ranges`;
- очистка масок морфологией и удаление мелких областей;
- объекты и площади через `skimage.measure.regionprops`;
- цветовая разметка, таблица `regions.csv`, статистика по классам.

Пороги HSV подобраны под конкретную сцену; для другого фото их нужно
калибровать и при необходимости передать файл через `--data`.

## Общие параметры CLI

| Аргумент | Назначение | По умолчанию |
|---|---|---|
| `variant` | `1`, `2`, `3` или `all` | `all` |
| `--config` | путь к JSON-конфигу | `configs/default.json` |
| `--out` | папка результатов | `out` |
| `--data-dir` | папка данных и кэша | `data` |
| `--data` | своё изображение вместо указанного в конфиге | — |
| `--set KEY=VALUE` | точечное переопределение конфига | — |

`--set` можно указывать несколько раз. Значение разбирается как JSON, поэтому
числа, списки и строки в кавычках поддерживаются:

```bash
python -m src all --set variant1.trials=5 --set variant1.median_kernel=7
```

## Результаты

```text
out/
├── variant1/
│   ├── before/                     # исходное и зашумлённое изображения
│   ├── after/                      # фильтры, CLAHE, совместный пайплайн
│   ├── noise_psnr_ssim.png         # зависимости PSNR/SSIM от уровня шума
│   ├── parameter_sweep.png         # чувствительность к параметрам фильтров
│   ├── filtering_comparison.png    # наглядное сравнение фильтров и контраста
│   ├── pipeline_comparison.png     # совместный пайплайн и усиление шума
│   ├── contrast_comparison.png     # histogram equalization vs CLAHE
│   ├── contrast_histograms.png
│   ├── metrics_noise_trials.csv
│   ├── metrics_noise_summary.csv
│   ├── metrics_parameter_sweep.csv
│   ├── metrics_contrast.csv
│   ├── metrics_pipeline.csv
│   └── summary.json
├── variant2/
│   ├── before/                     # исходное изображение и сырые маски
│   ├── after/                      # очищенные маски и контуры
│   ├── edge_masks.png
│   ├── contours_comparison.png
│   ├── object_counts.png
│   ├── area_distribution.png
│   ├── contours.csv
│   └── summary.json
└── variant3/
    ├── before/                     # исходное изображение и сырые маски
    ├── after/                      # очищенные маски и разметка
    ├── segmentation_steps.png
    ├── class_stats.png
    ├── regions.csv
    └── summary.json
```

Подробный отчёт с методикой, параметрами и обсуждением результатов —
в [`report.md`](report.md) (PDF-версия: [`report.pdf`](report.pdf)).

Пересобрать PDF из отчёта:

```bash
pip install -r requirements-report.txt
python scripts/build_report.py
```

## Структура проекта

```text
src/
├── __main__.py    # CLI
├── common.py      # загрузка данных, метрики, сохранение графиков
├── variant1.py    # шум и контраст
├── variant2.py    # границы и контуры
└── variant3.py    # цветовая сегментация
configs/default.json
data/              # пользовательские изображения и кэш skimage
out/               # результаты запуска
```
