# Локальный эксперимент TabM

Добавлен настоящий [TabM](https://github.com/yandex-research/tabm) из официального пакета `tabm==0.0.3` (Apache-2.0). Его несколько нейронных предикторов обучаются параллельно и используют общие веса. Веса создаются с нуля на данных задачи. Это отдельный эксперимент для CPU/CUDA, который пока не входит в автоматический конкурсный портфель или кнопки Alfa Lab. Измеренного улучшения на публичных задачах и проверки на RTX 3080 в этой поставке нет.

## Установка на Windows

Команды выполняются в папке актуального проекта. Окружение для нейросети отдельное, поэтому установка PyTorch не меняет окружение работающего приложения. `C:\Windows\py.exe` указан полностью из-за встречавшейся подмены команды `py`.

```powershell
& "C:\Windows\py.exe" -3.12 -m venv .venv-neural
if ($LASTEXITCODE -ne 0) { throw "venv failed" }
$python = (Resolve-Path .venv-neural\Scripts\python.exe).Path
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "base dependencies failed" }
& $python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
& $python -m pip install -r requirements.neural.txt
if ($LASTEXITCODE -ne 0) { throw "TabM installation failed" }
& $python experiment_tabular_neural.py --check
if ($LASTEXITCODE -ne 0) { throw "optional backend unavailable" }
& $python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Это официальный вариант установки PyTorch 2.5.1 для CUDA 12.1 из [архива команд PyTorch](https://pytorch.org/get-started/previous-versions/). Для CPU замените `cu121` на `cpu`. CUDA требует совместимого драйвера NVIDIA. `--check` проверяет пакеты, а фактическая доступность CUDA проверяется перед обучением. Если CUDA недоступна, запрос `--device cuda` завершится явной ошибкой и не переключится на CPU.

## Запуск

Найдите фактический `task.md` в своей папке данных, например:

```powershell
Get-ChildItem C:\aq-public -Filter task.md -Recurse | Select-Object -ExpandProperty FullName
```

Подставьте путь к нужной задаче. Каталог результата должен быть новым:

```powershell
$python = (Resolve-Path .venv-neural\Scripts\python.exe).Path
$output = Join-Path "C:\aq-neural" ("tabm-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
& $python experiment_tabular_neural.py "C:\aq-public\YOUR_TASK\task.md" --output $output --device cuda --time-budget 7200 --epochs 300 --patience 30 --members 16 --batch-size 512
if ($LASTEXITCODE -ne 0) { throw "TabM run incomplete; inspect $output\report.json" }
```

`7200` — верхняя граница, а не обещание двух часов нагрузки. Если внутренняя метрика не улучшается 30 эпох, поиск эпох заканчивается. Если память GPU заканчивается, уменьшите `--batch-size` до 256 или 128 и используйте новый каталог результата. Первый запуск лучше провести с `--epochs 10 --time-budget 600`; затем переходить к длинному эксперименту. Не запускайте его одновременно с очередью приложения: отдельный CLI не участвует в общей блокировке Alfa Lab.

## Что измеряется

1. Используется тот же автоматический выбор проверки, что у агента: группы дубликатов для классификации, подходящий временной столбец для регрессии, иначе случайное разбиение. Его можно явно задать через `--validation` и `--time-column`.
2. Внешние тренировочные строки делятся ещё раз по той же политике. Число эпох выбирается только на внутренней проверке.
3. Новый TabM обучается на всех внешних тренировочных строках ровно выбранное число эпох. Внешняя проверка не влияет на остановку, параметры или предобработку.
4. Внешняя метрика вычисляется один раз. `Macro F1` оценивает усреднённые вероятности с обычным argmax, `RMSE` — усреднённые числовые прогнозы.

Числовые медианы, масштабы и словари категорий считаются только на обучающей части каждого обучения. Пропуски и неизвестные категории обрабатываются явно. Лимит категорий защищает размер внутреннего one-hot представления. Проверка времени выполняется между мини-пакетами; это не жёсткий системный или VRAM-лимит. 35% времени отведено на выбор эпох, до 90% общего времени — на повторное обучение, остаток — на оценку и экспорт. Если повторное обучение не успело закончиться, отчёт `incomplete` не публикует валидную метрику.

## Файлы результата

- `report.json` — метрика, оба разбиения, отпечаток данных, версии пакетов, хеш реализации, выбранное число эпох, фактическое устройство и время.
- `holdout_predictions.csv` и `holdout_raw.npz` — внешние прогнозы и вероятности для последующего анализа.
- `test_predictions.csv` — прогнозы того же экземпляра, обученного только на внешней тренировочной части. Файл исходной задачи `submission.csv` не изменяется.
- `model/weights.pt` и `model/model.json` — веса на CPU, рецепт и предобработка. Это **модель внешнего тренировочного разбиения**, без дополнительного обучения на всех данных.

Для повторного прогнозирования своим сохранённым экземпляром:

```python
import pandas as pd
from research_lab.tabular_neural import TabMModel

model = TabMModel.load(r"C:\aq-neural\RUN\model", device="cpu")
raw = model.predict_raw(pd.read_csv(r"C:\aq-public\YOUR_TASK\inputs\test.csv", keep_default_na=False))
# raw: вероятности в порядке model.classes для классификации, числа для регрессии.
```

Загружайте только собственные доверенные артефакты. Для сравнения с LightGBM/ExtraTrees должны совпадать данные, внешнее разбиение и направление метрики. Высокая загрузка GPU сама по себе не доказывает улучшения. Новые пакеты пока не добавлены в `requirements.platform.txt`; готовности TabM к конкурсному контейнеру этот локальный эксперимент не утверждает.

## Проверки

`python -m unittest tests.test_tabular_neural -v` проверяет изоляцию предобработки, неизвестные категории, вложенную проверку, отсутствие изменения `submission.csv`, явный отказ CUDA и маленькие реальные CPU-обучения. Без дополнительных пакетов только реальные Torch-тесты пропускаются. CPU-проверки выполнены с PyTorch 2.5.1+cpu и TabM 0.0.3; GPU в среде разработки недоступен.
