# Windows: один длинный прогон принятого релиза

Этот порядок предназначен для **принятого релиза после раунда 4**, когда капитан
сообщил полный SHA и завершил native admission. Подготовка инструкции не означает,
что новый кандидат уже принят. Текущий статус —
[round4_integration_status.md](round4_integration_status.md).

Первый запуск: только публичная **регрессия**, CPU, восемь вычислительных потоков,
машина с 32 GB RAM, `solution.py --mode offline --time-budget 10500`. Задачи
выполняются последовательно. Не запускайте одновременно другой AutoML, обучение
в PyCharm, второй PowerShell-прогон или четыре задания на этой же машине.
RTX 3080 для этой проверки не требуется.

## Что уже проверено и чего пока нет

Исходный релиз e62 проверялся локально на 8 CPU / **8 GiB**, в том числе paired
900s: известный outer RMSE 44.569657210 → 39.471458256. Это development-проверка
на уже изученном public TRAIN, не метрика скрытого TEST. Она не доказывает работу
на 32 GB, Windows или полном бюджете 10 500 секунд. Результаты раунда 4 и точный
принятый SHA нужно брать из текущего status, а не подставлять старый e62 автоматически.

Сам длинный Windows-прогон этой инструкцией **не выполнен**. Доступность памяти,
телеметрия, завершение дерева процессов, full refit и соблюдение времени должны
быть подтверждены новыми артефактами. Отсутствующее измерение остаётся неизвестным.

При подготовке проверены компиляция встроенного Python, актуальные `--help`
трёх CLI, маленький процесс через существующий supervisor и полный wrapper
control flow с fake supervisor, настоящей сборкой ZIP и tiny CSV. Обучения и
API-запросов не было. PowerShell здесь отсутствует: синтаксис обёртки рассмотрен
статически, выполнение Windows/PowerShell 5.1 этими проверками не подтверждено.

## Подготовка окружения один раз

Откройте PowerShell **в корне проекта**, где лежат `solution.py` и
`prepare_data.py`. Активация virtualenv не нужна. Если `.venv` уже исправна и
содержит точные версии из `requirements.txt`, переустановка не требуется.
Для нового окружения:

```powershell
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Не удалось создать Python 3.12 venv" }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить pinned runtime" }
.\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw "Конфликт зависимостей" }
```

Это подготовка до измерения. Wrapper не вызывает `pip` и ничего не скачивает.
`requirements.txt` содержит runtime-подмножество официальных пакетов;
`requirements.platform.txt` сохраняет официальный список. Их копии, хеши и
фактические версии установленных пакетов попадут в evidence. Отсутствующий
AutoGluon или другой optional пакет нельзя выдавать за полное совпадение
окружения платформы. Не устанавливайте непроверенные версии ради завершения прогона.

## Основной блок: подготовка public и один запуск

Сначала перейдите на точный, уже принятый SHA обычным способом; рабочее дерево
должно быть чистым. Замените **только** значение `$qualificationRelease` на
полный SHA, сообщённый капитаном. Остальные строки можно вставить одним блоком.
Используются короткие пути вне репозитория; существующий каталог результата
никогда не очищается и не перезаписывается.

```powershell
$ErrorActionPreference = "Stop"
$qualificationRelease = "REPLACE_WITH_ACCEPTED_40_CHARACTER_SHA"
if ($qualificationRelease -notmatch '^[0-9a-fA-F]{40}$') { throw "Вставьте принятый полный SHA" }
$qualificationHead = (git rev-parse HEAD)
if ($LASTEXITCODE -ne 0) { throw "Откройте PowerShell в git-клоне проекта" }
if ($qualificationHead.Trim() -ne $qualificationRelease) { throw "HEAD не совпадает с принятым SHA" }

$qualificationPublic = "C:\aq-public"
$qualificationRoot = "C:\aq-r4-" + (Get-Date -Format "yyyyMMdd-HHmmss")
if (Test-Path -LiteralPath $qualificationRoot) { throw "Каталог результата уже существует" }

.\.venv\Scripts\python.exe prepare_data.py regression --data-dir $qualificationPublic
if ($LASTEXITCODE -ne 0) { throw "Ошибка подготовки public regression; обучение не запускать" }

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\round4_windows_qualification.ps1 -ReleaseCommit $qualificationRelease -PublicTask "$qualificationPublic\regression\task.md" -RunRoot $qualificationRoot -Python ".\.venv\Scripts\python.exe"
if ($LASTEXITCODE -ne 0) { throw "Проверка не завершилась чисто; сохраните каталог результата и остановите очередь" }

Write-Host "Результаты: $qualificationRoot"
```

`prepare_data.py` загружает только опубликованные организаторами файлы и
проверяет их размер, строки, колонки и ID. Это единственная сетевая подготовка
в основном блоке; уже скачанные файлы нужного размера используются повторно.
Для полностью офлайн-машины заранее перенесите готовый public-каталог и
вместо команды подготовки задайте путь к его `regression\task.md` в вызове wrapper.
Не копируйте прежние `.agent_runs`, `submission.csv` и модели в новый RunRoot.

`-ExecutionPolicy Bypass` применяется только к запускаемому процессу PowerShell,
системная политика не меняется. Python вызывается через **`.\.venv\Scripts\python.exe`**:
это точка, обратный слеш, затем `.venv`. `..venv` — другое, неверное имя.
При пути к интерпретатору с пробелами используется `& "полный путь\python.exe"`.

## Что делает wrapper

[round4_windows_qualification.ps1](../scripts/round4_windows_qualification.ps1)
проверяет Windows, Python 3.12, чистый принятый SHA, точные runtime pins и RAM
класса 32 GB. Он копирует в новый каталог только `task.md` и четыре исходных CSV,
фиксирует их SHA-256, собирает архив существующим `build_submission.py --output`
и запускает именно распакованный **flat ZIP**, а не другую копию исходников.
Архив содержит официальный `requirements.platform.txt` под именем `requirements.txt`.

Для запуска используется существующий `research_lab.process.supervise`.
Он передаёт дочернему процессу очищенное окружение без API credentials,
ограничивает native threads восемью, скрывает CUDA, сохраняет ограниченные логи,
измеряет RSS дерева и завершает только принадлежащие ему процессы. Общий lock
предотвращает одновременные запуски этого wrapper и совместимых laboratory runner
для одного пользователя. Он не обнаруживает произвольное чужое обучение: его
нужно закончить до запуска. Wrapper не вызывает `--check-api`, не читает `.env`
и не печатает значения переменных окружения или ключи.

Запрошенный sampled tree-RSS threshold — **28 GiB**, штатный порог агента.
32 GB — характеристика машины, а не право занять всю её память. В evidence
сохраняются фактические total/available RAM. RSS может учитывать общие страницы
несколько раз; polling не является жёстким ограничением памяти ОС и не доказывает
абсолютный пик. Восемь потоков также не заменяют аппаратную квоту восьми CPU.

У native CLI ровно **10 500 секунд**. Внешний watchdog **10 530 секунд** нужен
для остановки зависания около запуска/завершения и не добавляет разрешённого
времени обучению. В `qualification.json` отдельно проверяются native
`time_budget_seconds`, `elapsed_seconds` и exit code; внешний wall clock и
cleanup сохраняются в `execution.json`. Сам факт завершения раньше 10 530 секунд
не доказывает соблюдение native deadline. Неясное превышение 10 500 секунд или
неполные native timings требуют разбора, а не отметки «квалификация пройдена».

## Разбор результата до следующей задачи

В RunRoot останутся `plan.json`, оба requirements, `build.log`, собранный
`solution.zip`, `execution.json`, `qualification.json`, ограниченные
`execution\stdout.txt` / `stderr.txt`, полный native `task\.agent_runs\…\run.json`,
receipts и `task\submission.csv`. Свежие CSV, модельные артефакты и receipts
не удаляются. Сборка не отправляет ZIP организаторам.

Успех команды означает только прохождение перечисленных автоматических
mechanism checks. Итоговый статус всегда **`requires_review`**. До продолжения
нужен разбор следующих фактов:

1. Точный SHA, pins, flat ZIP hash, неизменность исходных CSV и runtime sources.
2. Root exit 0, watchdog/memory stop отсутствуют; resources доступны и owned
   cleanup complete. `null`/unavailable не превращать в ноль или успешную проверку.
3. Native API `attempts=0`, `responses=0`; это фактические счётчики отчёта.
4. INNER-кандидат реально выполнен, outer решение и frozen recipe согласованы;
   full refit использует все TRAIN, публикует predictions для исходного TEST,
   хеши checkpoint/receipts/output совпадают. Валидная форма CSV сама по себе
   не доказывает full refit. Target-prior/emergency fallback не считать
   квалификацией новой модели, даже при exit 0.
5. Deadline подтверждён native timing и внешним execution; RSS/cleanup наблюдаемы.
   Известный public outer score записать отдельно, без заявления о hidden TEST.

Только после этого планировать следующую **одну** serial-задачу в новом RunRoot.
Классификация требует отдельного согласованного прогона: данный wrapper специально
принимает только RMSE-регрессию. Не расширяйте очередь автоматически после успеха.

Для анализа сначала достаточно `plan.json`, `execution.json`,
`qualification.json`, native `run.json` и ограниченных логов. При неуспехе
сохраните текущие артефакты; повторный запуск получает новый каталог. Причину
`source drift`, timeout, memory stop, unavailable RSS или fallback разбирают
до продолжения. Эти файлы содержат пути машины и сведения о runtime;
общий чат организаторов не является местом для публикации кода и рецептов.
