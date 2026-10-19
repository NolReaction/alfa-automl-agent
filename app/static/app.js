/* Alfa Lab: a local, read-only view of existing task files and agent reports. */
"use strict";
const $ = (query, root = document) => root.querySelector(query);
const $$ = (query, root = document) => [...root.querySelectorAll(query)];
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (value, digits = 0) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("ru-RU", { maximumFractionDigits: digits })
    : "—";
const percent = (value) => `${number(value, 2)} %`;
const duration = (value) => {
  if (!Number.isFinite(value)) return "—";
  if (value < 60) return `${number(value, 1)} с`;
  const m = Math.floor(value / 60);
  return m < 60
    ? `${m} мин ${Math.floor(value % 60)} с`
    : `${Math.floor(m / 60)} ч ${m % 60} мин`;
};
const date = (value) => {
  const d = new Date(value);
  return value && Number.isFinite(d.getTime())
    ? d.toLocaleString("ru-RU", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Время не записано";
};
const ICONS = {
  data: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M9 9v12"/>',
  activity: '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
  check:
    '<path d="m7 12 3 3 7-7"/><rect x="3" y="3" width="18" height="18" rx="5"/>',
  terminal:
    '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="m7 9 3 3-3 3m6 0h4"/>',
  shield:
    '<path d="M12 3 4 6v6c0 4 8 9 8 9s8-5 8-9V6l-8-3Z"/><path d="m8 12 3 3 5-6"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 4 2c-1 .7-1.5 1-1.5 2M12 16h.01"/>',
  refresh:
    '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M6 6a8 8 0 0 1 13 3M18 18A8 8 0 0 1 5 15"/>',
  file: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6Z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
  rows: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M3 14h18M8 4v16"/>',
  test: '<path d="M9 3v6l-5 9a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-5-9V3M7 3h10M7 15h10"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
  missing: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5M12 16h.01"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  code: '<path d="m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  download: '<path d="M12 3v12m-4-4 4 4 4-4M4 15v5h16v-5"/>',
  chart: '<path d="M4 3v17h17M8 15l4-5 4 3 5-7"/>',
  arrowdown: '<path d="M12 4v16m-6-6 6 6 6-6"/>',
  arrowup: '<path d="M12 20V4m-6 6 6-6 6 6"/>',
  copy: '<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
  folder:
    '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
};
const icon = (name) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.file}</svg>`;
const badge = (text, color = "") =>
  `<span class="badge ${color}">${esc(text)}</span>`;
const stat = (label, value, caption, symbol, compact = false) =>
  `<div class="stat-card"><div class="stat-top"><span>${esc(label)}</span>${icon(symbol)}</div><div class="stat-value${compact ? " compact" : ""}">${esc(value)}</div><div class="stat-caption">${esc(caption)}</div></div>`;
const loading = (text) =>
  `<div class="initial-loading"><span class="spinner"></span><p>${esc(text || "Загружаем данные…")}</p></div>`;
const inlineLoading = () =>
  '<div class="inline-loading"><span class="spinner"></span><p>Загрузка…</p></div>';
const state = {
  tasks: [],
  task: null,
  view: "data",
  profiles: new Map(),
  profile: null,
  generation: 0,
  columnGeneration: 0,
  runGeneration: 0,
  fileGeneration: 0,
  selectedColumn: null,
  runs: [],
  selectedRun: null,
  runDetail: null,
  experiment: null,
  fileKind: "code",
  pollBusy: false,
  lastRunFingerprint: null,
};
const main = $("#main");
function apiBase(task = state.task) {
  return `/api/tasks/${encodeURIComponent(task)}`;
}
async function api(path) {
  const response = await fetch(path, {
    cache: "no-store",
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(
      "Сервер вернул непонятный ответ. Проверьте, что приложение запущено.",
    );
  }
  if (!response.ok) throw new Error(body.error || `Ошибка ${response.status}`);
  return body;
}
function taskInfo() {
  return state.tasks.find((task) => task.id === state.task);
}
function heading(eyebrow, title, description, actions = true) {
  return `<div class="page-heading"><div><span class="eyebrow">${esc(eyebrow)}</span><h1>${esc(title)}</h1><p class="page-description">${esc(description)}</p></div>${actions ? `<div class="heading-actions"><button class="button" data-action="refresh" title="Обновить данные">${icon("refresh")}<span class="button-label">Обновить</span></button><button class="button" data-action="help" title="Команды запуска">${icon("terminal")}<span class="button-label">Команды</span></button></div>` : ""}</div>`;
}
function taskStrip(profile) {
  return `<div class="summary-strip">${badge(profile?.task_type === "classification" ? "Классификация" : profile?.task_type === "regression" ? "Регрессия" : "ML-задача", "red")}${badge(profile?.target || taskInfo()?.target || "Таргет не определён")}<span class="subtle-text">${esc(taskInfo()?.name || "")}</span></div>`;
}
function empty(title, message, symbol = "folder", action = true) {
  return `<section class="empty-state"><span class="empty-icon">${icon(symbol)}</span><h2>${esc(title)}</h2><p>${esc(message)}</p>${action ? '<button class="button" data-action="help">Команды запуска</button>' : ""}</section>`;
}
function errorView(error, title = "Не удалось загрузить данные") {
  return `<section class="empty-state error-panel"><span class="empty-icon">${icon("missing")}</span><h2>${esc(title)}</h2><p>${esc(error.message || error)}</p><button class="button" data-action="refresh">Попробовать ещё раз</button></section>`;
}
function warningList(values) {
  return values?.length
    ? `<div class="warning-box"><ul>${values.map((v) => `<li>${esc(v)}</li>`).join("")}</ul></div>`
    : "";
}
function typeName(type) {
  return (
    {
      numeric: "Числовой",
      category: "Категория",
      binary: "Бинарный",
      unknown: "Неизвестный",
    }[type] ||
    type ||
    "Неизвестный"
  );
}
function chart(distribution, label = "Распределение") {
  const bins = Array.isArray(distribution?.bins) ? distribution.bins : [];
  if (!bins.length)
    return `<div class="chart-empty">${icon("chart")}<span>Нет значений для распределения</span></div>`;
  const max = Math.max(1, ...bins.map((b) => Number(b.count) || 0));
  if (distribution.kind === "histogram") {
    return `<div class="chart" role="img" aria-label="${esc(label)}"><div class="histogram">${bins.map((b) => `<div class="hist-bin" style="--height:${(Math.max(0, Number(b.count) || 0) / max) * 100}%" title="${esc(b.label)}: ${number(b.count)}" data-tooltip="${esc(b.label)} · ${number(b.count)}"><span class="sr-only">${esc(b.label)}: ${number(b.count)}</span></div>`).join("")}</div><div class="chart-axis"><span>${esc(bins[0].label)}</span><span>${esc(bins.at(-1).label)}</span></div></div>`;
  }
  return `<div class="bar-chart" role="img" aria-label="${esc(label)}">${bins.map((b) => `<div class="bar-row"><span class="bar-label" title="${esc(b.label)}">${esc(b.label)}</span><div class="bar-track"><div class="bar-fill" style="--width:${(Math.max(0, Number(b.count) || 0) / max) * 100}%"></div></div><span class="bar-value">${number(b.count)}</span></div>`).join("")}</div>`;
}
function missingChart(profile) {
  const rows = (profile.missing || [])
    .filter((item) => item.count > 0)
    .slice(0, 10);
  if (!rows.length)
    return `<div class="chart-empty">${icon("check")}<div><strong>Пропусков нет</strong><span>Все ячейки обучающей выборки заполнены</span></div></div>`;
  const max = Math.max(1, ...rows.map((r) => r.percent));
  return `<div class="bar-chart" role="img" aria-label="Доля пропусков по столбцам">${rows.map((r) => `<div class="bar-row"><span class="bar-label" title="${esc(r.name)}">${esc(r.name)}</span><div class="bar-track"><div class="bar-fill" style="--width:${(Math.max(0, r.percent) / max) * 100}%"></div></div><span class="bar-value" title="${number(r.count)} ячеек">${number(r.percent, 1)} %</span></div>`).join("")}</div><p class="chart-footnote">${rows.length} столбцов с наибольшей долей пропусков · train.csv</p>`;
}
async function loadTasks() {
  const generation = ++state.generation;
  main.innerHTML = loading("Ищем локальные задачи…");
  try {
    const data = await api("/api/tasks");
    if (generation !== state.generation) return;
    state.tasks = Array.isArray(data.tasks) ? data.tasks : [];
    const select = $("#task-select");
    if (!state.tasks.length) {
      state.task = null;
      select.innerHTML = "<option>Задач пока нет</option>";
      select.disabled = true;
      main.innerHTML =
        heading(
          "ОБЗОР ДАННЫХ",
          "Начнём с данных",
          "Рабочее пространство готово к первой задаче.",
        ) +
        empty(
          "Задачи ещё не загружены",
          "Скачайте публичные данные командой prepare_data.py all. После загрузки нажмите «Обновить».",
          "folder",
        );
      return;
    }
    if (!state.tasks.some((t) => t.id === state.task))
      state.task =
        state.tasks.find((t) => t.available)?.id || state.tasks[0].id;
    select.innerHTML = state.tasks
      .map(
        (t) =>
          `<option value="${esc(t.id)}">${esc(t.name)}${t.available ? "" : " · нет данных"}</option>`,
      )
      .join("");
    select.value = state.task;
    select.disabled = false;
    await renderView();
  } catch (error) {
    if (generation === state.generation)
      main.innerHTML = errorView(
        error,
        "Не удалось открыть рабочее пространство",
      );
  }
}
async function renderView() {
  const generation = ++state.generation;
  state.columnGeneration++;
  state.runGeneration++;
  state.fileGeneration++;
  state.lastRunFingerprint = null;
  const names = { data: "Данные", runs: "Запуски", result: "Результат" };
  $$(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === state.view);
    if (button.dataset.view === state.view)
      button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("#crumb-name").textContent = names[state.view];
  if (!state.task) {
    await loadTasks();
    return;
  }
  main.innerHTML = loading(
    state.view === "data"
      ? "Считаем статистику по таблицам…"
      : "Читаем локальные отчёты…",
  );
  try {
    if (state.view === "data") await loadData(generation);
    else if (state.view === "runs") await loadRuns(generation);
    else await loadResult(generation);
  } catch (error) {
    if (generation === state.generation) main.innerHTML = errorView(error);
  }
}
async function loadData(generation) {
  const task = state.task;
  const profile =
    state.profiles.get(task) || (await api(`${apiBase(task)}/profile`));
  if (generation !== state.generation) return;
  state.profiles.set(task, profile);
  state.profile = profile;
  if (!profile.columns?.some((c) => c.name === state.selectedColumn))
    state.selectedColumn =
      profile.columns?.find((c) => c.role === "feature")?.name ||
      profile.columns?.[0]?.name;
  const types = profile.type_counts || {};
  main.innerHTML =
    heading(
      "ОБЗОР ДАННЫХ",
      "Исследование данных",
      "Изучите структуру выборки, распределения и признаки перед запуском экспериментов.",
    ) +
    taskStrip(profile) +
    `<div class="stats-grid">${stat("Обучающая выборка", number(profile.train_rows), "строк в train.csv", "rows")}${stat("Тестовая выборка", number(profile.test_rows), "строк в test.csv", "test")}${stat("Входные признаки", number(profile.feature_count), "без идентификатора и таргета", "layers")}${stat("Пропуски в train", percent(profile.missing_percent), `${number(profile.missing_cells)} пустых ячеек`, "missing")}</div>` +
    warningList(profile.warnings) +
    `<div class="chart-grid"><section class="panel"><div class="panel-head"><div><h2>Распределение таргета</h2><p>${esc(profile.target)} · обучающая выборка</p></div><span class="panel-label"><span class="mini-dot"></span> TRAIN</span></div><div class="panel-body">${chart(profile.target_distribution, "Распределение таргета " + profile.target)}${profile.target_distribution?.missing_count ? `<p class="chart-footnote">Пропущено: ${number(profile.target_distribution.missing_count)}</p>` : ""}</div></section><section class="panel"><div class="panel-head"><div><h2>Пропуски в данных</h2><p>Доля пустых ячеек по столбцам</p></div>${badge(`${number(profile.missing_percent, 2)} %`, "gray")}</div><div class="panel-body">${missingChart(profile)}</div></section></div>` +
    `<section class="panel feature-section"><div class="panel-head"><div><h2>Исследователь признаков</h2><p>Выберите столбец, чтобы увидеть его статистику</p></div><div class="type-line"><span class="type-dot"></span>${number(types.numeric)} числ.<span class="type-dot category"></span>${number(types.category)} кат.<span class="type-dot binary"></span>${number(types.binary)} бин.</div></div><div class="feature-layout"><div class="feature-list-pane"><div class="feature-search search-wrap">${icon("search")}<input id="feature-search" class="search-input" type="search" placeholder="Найти столбец…" aria-label="Найти столбец по имени"></div><div id="feature-list" class="feature-list"></div></div><div class="feature-detail" id="feature-detail">${inlineLoading()}</div></div></section>` +
    `<section class="panel"><div class="panel-head"><div><h2>Предпросмотр таблицы</h2><p>Первые ${number(profile.preview?.rows?.length || 0)} строк train.csv</p></div>${badge("TRAIN", "gray")}</div><div class="table-toolbar"><div class="search-wrap">${icon("search")}<input id="table-search" type="search" class="search-input" placeholder="Показать столбцы по имени…" aria-label="Фильтр столбцов таблицы"></div><span class="subtle-text" id="column-count"></span></div><div class="table-wrap" id="preview-table"></div><div class="table-footer"><span>Предпросмотр ограничен 100 строками</span><span>Исходные CSV доступны только на вашем компьютере</span></div></section>` +
    `<details class="panel description-box" style="margin-top:23px"><summary>Описание задачи · task.md</summary><pre id="task-description">Откройте описание, чтобы загрузить текст.</pre></details>`;
  renderFeatures("");
  renderTable("");
  $("#feature-search").addEventListener("input", (event) =>
    renderFeatures(event.target.value),
  );
  $("#table-search").addEventListener("input", (event) =>
    renderTable(event.target.value),
  );
  $(".description-box").addEventListener("toggle", async (event) => {
    if (event.target.open) {
      const text = $("#task-description");
      if (text.dataset.loaded) return;
      text.textContent = "Загружаем описание…";
      try {
        const result = await api(`${apiBase(task)}/description`);
        if (generation === state.generation) {
          text.textContent = result.text;
          text.dataset.loaded = "true";
        }
      } catch (error) {
        if (generation === state.generation) text.textContent = error.message;
      }
    }
  });
  if (state.selectedColumn) await loadColumn(state.selectedColumn);
}
function renderFeatures(query) {
  const list = $("#feature-list");
  if (!list) return;
  const columns = (state.profile?.columns || []).filter((c) =>
    c.name.toLocaleLowerCase("ru").includes(query.toLocaleLowerCase("ru")),
  );
  list.innerHTML =
    columns
      .map(
        (c) =>
          `<button class="feature-item ${c.name === state.selectedColumn ? "selected" : ""}" data-column="${esc(c.name)}" aria-pressed="${c.name === state.selectedColumn}"><span class="feature-name">${esc(c.name)}</span><span class="feature-type">${c.role === "target" ? "Таргет" : c.role === "id" ? "ID" : esc(typeName(c.dtype))}</span></button>`,
      )
      .join("") || '<div class="inline-loading">Столбцы не найдены</div>';
}
async function loadColumn(name) {
  state.selectedColumn = name;
  renderFeatures($("#feature-search")?.value || "");
  const generation = ++state.columnGeneration,
    task = state.task,
    viewGeneration = state.generation;
  $("#feature-detail").innerHTML = inlineLoading();
  try {
    const value = await api(
      `${apiBase(task)}/column?name=${encodeURIComponent(name)}`,
    );
    if (
      generation !== state.columnGeneration ||
      viewGeneration !== state.generation
    )
      return;
    const stats = value.stats || {};
    $("#feature-detail").innerHTML =
      `<div class="feature-title"><h3>${esc(value.name)}</h3>${badge(typeName(value.dtype), "gray")}</div><p class="feature-description">${esc(value.description || "Описание столбца не указано в словаре данных.")}</p><div class="feature-stat-grid"><div><span>Уникальных значений</span><strong>${number(value.unique_count)}</strong></div><div><span>Пропусков</span><strong>${number(value.missing_count)}</strong></div><div><span>Доля пропусков</span><strong>${percent(value.missing_percent)}</strong></div></div>${Object.keys(stats).length ? `<div class="numeric-stats"><span>Мин.<b>${number(stats.min, 3)}</b></span><span>Макс.<b>${number(stats.max, 3)}</b></span><span>Медиана<b>${number(stats.median, 3)}</b></span><span>Среднее<b>${number(stats.mean, 3)}</b></span></div>` : ""}${chart(value.distribution, "Распределение признака " + value.name)}`;
  } catch (error) {
    if (
      generation === state.columnGeneration &&
      viewGeneration === state.generation
    )
      $("#feature-detail").innerHTML =
        `<div class="warning-box">${esc(error.message)}</div>`;
  }
}
function renderTable(query) {
  const preview = state.profile?.preview || { columns: [], rows: [] };
  const columns = preview.columns
    .map((name, index) => ({ name, index }))
    .filter((c) =>
      c.name.toLocaleLowerCase("ru").includes(query.toLocaleLowerCase("ru")),
    );
  $("#column-count").textContent =
    `${columns.length} из ${preview.columns.length} столбцов`;
  $("#preview-table").innerHTML = columns.length
    ? `<table aria-label="Первые строки обучающей выборки"><thead><tr><th scope="col">#</th>${columns.map((c) => `<th scope="col" title="${esc(c.name)}">${esc(c.name)}</th>`).join("")}</tr></thead><tbody>${preview.rows
        .map(
          (row, index) =>
            `<tr><td>${index + 1}</td>${columns
              .map((c) => {
                const value = row[c.index];
                return `<td class="${value === "" ? "empty-cell" : ""}" title="${esc(value)}">${value === "" ? "пусто" : esc(value)}</td>`;
              })
              .join("")}</tr>`,
        )
        .join("")}</tbody></table>`
    : '<div class="inline-loading">Нет столбцов с таким именем</div>';
}
const STATUS = {
  running: ["В работе", "red"],
  completed: ["Завершён", "green"],
  fallback: ["Резервный результат", "amber"],
  failed: ["Ошибка", "red"],
  invalid: ["Отчёт недоступен", "amber"],
};
function statusBadge(status) {
  const v = STATUS[status] || ["Неизвестный статус", "gray"];
  return badge(v[0], v[1]);
}
async function loadRuns(generation, quiet = false) {
  const task = state.task;
  const data = await api(`${apiBase(task)}/runs`);
  if (generation !== state.generation) return;
  state.runs = Array.isArray(data.runs) ? data.runs : [];
  if (!state.runs.length) {
    main.innerHTML =
      heading(
        "НАБЛЮДЕНИЕ",
        "Запуски и эксперименты",
        "Смотрите, что делает агент, и разбирайте результаты каждого шага.",
      ) +
      taskStrip(taskInfo()) +
      empty(
        "Первый запуск ещё впереди",
        "Когда агент начнёт исследование, здесь появятся его журнал, код экспериментов и вывод. Этот экран обновляется автоматически.",
        "activity",
      );
    return;
  }
  if (!state.runs.some((r) => r.id === state.selectedRun))
    state.selectedRun = state.runs[0].id;
  if (!quiet || !$("#run-list"))
    main.innerHTML =
      heading(
        "НАБЛЮДЕНИЕ",
        "Запуски и эксперименты",
        "Код, события и результаты исполнения в одном рабочем пространстве.",
      ) +
      taskStrip(taskInfo()) +
      `<div class="runs-layout"><section class="panel run-list-panel"><div class="panel-head"><h2>История запусков</h2><span class="subtle-text" id="run-count"></span></div><div class="run-list" id="run-list"></div></section><div id="run-detail" class="run-detail">${inlineLoading()}</div></div>`;
  renderRunList();
  await loadRun(state.selectedRun, quiet);
}
function renderRunList() {
  $("#run-count").textContent = String(state.runs.length);
  $("#run-list").innerHTML = state.runs
    .map(
      (r) =>
        `<button class="run-item ${r.id === state.selectedRun ? "selected" : ""}" data-run="${esc(r.id)}" aria-pressed="${r.id === state.selectedRun}"><div class="run-item-top"><span class="run-id">${esc(r.id.slice(0, 8))}</span>${statusBadge(r.status)}</div><div class="run-date">${esc(date(r.started_at))}</div><div class="run-item-bottom"><span>${number(r.experiments)} исполнений</span><span>${duration(r.elapsed_seconds)}</span></div>${r.error ? `<div class="event-body">${esc(r.error)}</div>` : ""}</button>`,
    )
    .join("");
}
function usageView(usage) {
  if (!usage || usage.state === "unknown")
    return "<strong>Токены: неизвестно</strong><span>API не вернул сведения о расходе.</span>";
  if (usage.state === "none")
    return "<strong>API не вызывался</strong><span>Запросов в этом запуске не было.</span>";
  return `<strong>${number(usage.total_tokens)} токенов${usage.state === "partial" ? " · частичный учёт" : ""}</strong><span class="usage-separator">/</span><span>${number(usage.prompt_tokens)} вход · ${number(usage.completion_tokens)} выход</span>${usage.state === "partial" ? `<span>Учёт по ${number(usage.reported_responses)} из ${number(usage.responses)} ответов.</span>` : ""}`;
}
function eventView(event, index) {
  const result = event.result || {};
  let title,
    body = "",
    color = "";
  switch (event.kind) {
    case "existing_output_preserved":
      title = "Сохранён стартовый результат";
      body =
        "Существующий CSV прошёл проверку формата и сохранён как резервный.";
      color = "success";
      break;
    case "baseline":
      title = "Обучение стартовой модели";
      body = event.valid
        ? "Стартовый CSV прошёл проверку формата."
        : "Стартовый результат недоступен. Подробности в выводе эксперимента.";
      color = event.valid ? "success" : "warn";
      break;
    case "experiment":
      title = `Эксперимент · шаг ${event.step ?? "—"}, вызов ${event.tool_index ?? "—"}`;
      color =
        result.error || result.exit_code !== 0 || result.timed_out
          ? "error"
          : result.output_check?.valid
            ? "success"
            : "warn";
      body =
        result.error ||
        (result.timed_out
          ? "Превышен лимит времени."
          : `Код завершился с кодом ${result.exit_code ?? "—"}. ${result.output_check?.valid ? "CSV прошёл проверку формата." : "CSV не прошёл проверку формата."}`);
      if (result.previous_output_restored)
        body += " Предыдущий CSV восстановлен.";
      break;
    case "output_restored":
      title = "Восстановлен предыдущий CSV";
      body = event.valid
        ? "Резервный файл прошёл проверку формата."
        : "После восстановления проверка не пройдена.";
      color = event.valid ? "warn" : "error";
      break;
    case "api_error":
      title = "Ошибка обращения к API";
      body = event.error || "Ответ не получен.";
      color = "error";
      break;
    case "failed":
      title = "Запуск завершился с ошибкой";
      body = event.error || event.reason || "";
      color = "error";
      break;
    default:
      title = event.kind || "Событие";
      body = event.error || event.reason || "";
  }
  return `<div class="timeline-event ${color}"><span class="event-dot"></span><div class="event-header"><span class="event-title">${esc(title)}</span><span class="event-index">${String(index + 1).padStart(2, "0")}</span></div><p class="event-body">${esc(body)}</p>${result.output_check?.errors?.length ? `<p class="event-details">${result.output_check.errors.map(esc).join(" · ")}</p>` : ""}</div>`;
}
async function loadRun(id, quiet = false) {
  const generation = ++state.runGeneration,
    viewGeneration = state.generation,
    task = state.task;
  const changed = state.selectedRun !== id;
  state.selectedRun = id;
  renderRunList();
  if (changed) {
    state.experiment = null;
    state.fileKind = "code";
    state.lastRunFingerprint = null;
    state.runDetail = null;
    state.fileGeneration++;
    $("#run-detail").innerHTML = inlineLoading();
  }
  try {
    const detail = await api(`${apiBase(task)}/runs/${encodeURIComponent(id)}`);
    if (
      generation !== state.runGeneration ||
      viewGeneration !== state.generation
    )
      return;
    const fingerprint = JSON.stringify(detail);
    if (quiet && fingerprint === state.lastRunFingerprint) {
      const el = $("#last-refresh");
      if (el)
        el.textContent = `Проверено в ${new Date().toLocaleTimeString("ru-RU")}`;
      await loadRunFile(true);
      return;
    }
    const savedScroll = quiet
      ? {
          timeline: $(".timeline")?.scrollTop || 0,
          codeTop: $("#code-content")?.scrollTop || 0,
          codeLeft: $("#code-content")?.scrollLeft || 0,
          focus: document.activeElement?.id,
        }
      : null;
    state.lastRunFingerprint = fingerprint;
    state.runDetail = detail;
    const r = detail.report || {},
      files = detail.files || [],
      experiments = [...new Set(files.map((f) => f.experiment))].sort(
        (a, b) => a - b,
      );
    if (!experiments.includes(state.experiment))
      state.experiment = experiments.at(-1) ?? null;
    const baselineNames = {
      existing_output: "Готовый CSV",
      trained: "Модель обучена",
      unavailable: "Недоступна",
      not_started: "Не запускалась",
      skipped_for_time: "Недостаточно времени",
    };
    const reasonNames = {
      model_finished: "Модель завершила исследование",
      time_limit: "Достигнут лимит времени",
      step_limit: "Достигнут лимит шагов",
      api_error: "Ошибка API",
      truncated_response: "Ответ модели был обрезан",
    };
    $("#run-detail").innerHTML =
      `<div class="run-detail-header"><div><h2>Запуск ${esc(id.slice(0, 8))}</h2><p>${esc(date(r.started_at))} · ${esc(r.model || "Модель не указана")}</p><div class="refresh-time"><span class="connection-dot"></span><span id="last-refresh">Обновление каждые 4 секунды</span></div></div>${statusBadge(r.status)}</div><div class="stats-grid">${stat("Время запуска", duration(r.elapsed_seconds), "по последнему отчёту", "clock", true)}${stat("Исполнений", number(r.experiments), `${number(r.successful_experiments)} с корректным CSV`, "code")}${stat("Запросов API", number(detail.usage?.attempts ?? r.api?.attempts), "попыток обращения", "activity")}${stat("Формат CSV", r.output_check?.valid === true ? "Корректен" : r.output_check?.valid === false ? "Ошибка" : "Не проверен", "не оценка ML-качества", "check", true)}</div><div class="usage-row">${usageView(detail.usage)}</div>${warningList(detail.warnings)}<section class="panel"><div class="panel-head"><div><h2>Журнал событий</h2><p>События в порядке выполнения</p></div>${badge(`${number(r.events?.length || 0)} событий`, "gray")}</div><div class="timeline">${r.events?.length ? r.events.map(eventView).join("") : '<p class="event-body">События ещё не записаны.</p>'}</div></section><section class="panel"><div class="panel-head"><div><h2>Код и вывод</h2><p>Файлы выбранного эксперимента</p></div>${experiments.length ? `<label class="sr-only" for="experiment-select">Выбрать эксперимент</label><select id="experiment-select" class="file-select">${experiments.map((value) => `<option value="${value}"${state.experiment === value ? " selected" : ""}>${value === 0 ? "Стартовая модель" : `Эксперимент ${value}`}</option>`).join("")}</select>` : ""}</div><div class="code-tabs" role="tablist" aria-label="Файл эксперимента">${[
        ["code", "Python"],
        ["stdout", "Вывод"],
        ["stderr", "Ошибки"],
      ]
        .map(
          ([kind, label]) =>
            `<button class="code-tab ${state.fileKind === kind ? "active" : ""}" data-file-kind="${kind}" role="tab" aria-selected="${state.fileKind === kind}">${label}</button>`,
        )
        .join(
          "",
        )}</div><pre class="code-content" id="code-content" role="tabpanel">${experiments.length ? "Загружаем файл…" : "Файлы экспериментов ещё не появились."}</pre><div class="code-footnote" id="code-footnote">Исходный текст файла · только просмотр</div></section><p class="run-note">Стартовый результат: ${esc(baselineNames[r.baseline] || r.baseline || "нет сведений")}.${r.reason ? ` Причина завершения: ${esc(reasonNames[r.reason] || r.reason)}.` : ""} Статус «Завершён» подтверждает исполнение и формат CSV. Он не подтверждает улучшение метрики.</p>`;
    $("#experiment-select")?.addEventListener("change", (event) => {
      state.experiment = Number(event.target.value);
      loadRunFile();
    });
    if (experiments.length) await loadRunFile(quiet);
    if (
      savedScroll &&
      generation === state.runGeneration &&
      viewGeneration === state.generation
    ) {
      const timeline = $(".timeline"),
        code = $("#code-content");
      if (timeline) timeline.scrollTop = savedScroll.timeline;
      if (code) {
        code.scrollTop = savedScroll.codeTop;
        code.scrollLeft = savedScroll.codeLeft;
      }
      if (savedScroll.focus === "experiment-select")
        $("#experiment-select")?.focus({ preventScroll: true });
    }
  } catch (error) {
    if (
      generation === state.runGeneration &&
      viewGeneration === state.generation
    )
      $("#run-detail").innerHTML = errorView(
        error,
        "Не удалось прочитать запуск",
      );
  }
}
async function loadRunFile(quiet = false) {
  const generation = ++state.fileGeneration,
    viewGeneration = state.generation,
    run = state.selectedRun,
    task = state.task;
  $$(".code-tab").forEach((button) => {
    button.classList.toggle(
      "active",
      button.dataset.fileKind === state.fileKind,
    );
    button.setAttribute(
      "aria-selected",
      String(button.dataset.fileKind === state.fileKind),
    );
  });
  const file = state.runDetail?.files?.find(
    (f) => f.experiment === state.experiment && f.kind === state.fileKind,
  );
  const output = $("#code-content"),
    foot = $("#code-footnote");
  if (!file) {
    output.textContent = "Этот файл пока не создан.";
    foot.textContent = "Исходный текст файла · только просмотр";
    return;
  }
  const scroll = { top: output.scrollTop, left: output.scrollLeft };
  if (!quiet) output.textContent = "Загружаем файл…";
  try {
    const data = await api(
      `${apiBase(task)}/runs/${encodeURIComponent(run)}/file?name=${encodeURIComponent(file.name)}`,
    );
    if (
      generation !== state.fileGeneration ||
      viewGeneration !== state.generation ||
      run !== state.selectedRun
    )
      return;
    const nextText =
      data.text ||
      (state.fileKind === "stderr" ? "Вывод ошибок пуст." : "Файл пуст.");
    if (output.textContent !== nextText) output.textContent = nextText;
    if (quiet) {
      output.scrollTop = scroll.top;
      output.scrollLeft = scroll.left;
    }
    foot.textContent = `${data.name}${data.truncated ? " · показан ограниченный фрагмент" : " · полный файл"}`;
  } catch (error) {
    if (
      generation === state.fileGeneration &&
      viewGeneration === state.generation
    )
      output.textContent = error.message;
  }
}
async function loadResult(generation) {
  const data = await api(`${apiBase()}/result`);
  if (generation !== state.generation) return;
  const b = data.baseline,
    s = data.submission || {},
    metric = b?.metric,
    split = b?.split || {},
    timing = b?.timing_seconds || {};
  const metricValid =
    metric && typeof metric.value === "number" && Number.isFinite(metric.value);
  let baseline;
  if (!b)
    baseline = `<div class="panel-body"><div class="chart-empty">${icon("chart")}<span>${data.baseline_error ? "Отчёт baseline не удалось прочитать" : "Отчёта baseline пока нет"}</span></div>${data.baseline_error ? `<div class="warning-box">${esc(data.baseline_error)}</div>` : '<p class="run-note">После запуска offline_baseline.py здесь появятся модель, локальная метрика и параметры проверки.</p>'}</div>`;
  else
    baseline = `<div class="panel-body"><div class="metric-main"><strong class="metric-number">${metricValid ? number(metric.value, 6) : "—"}</strong><span class="metric-name">${esc(metric?.name || "Метрика не указана")}</span></div><p class="direction-note">${icon(metric?.higher_is_better === true ? "arrowup" : metric?.higher_is_better === false ? "arrowdown" : "chart")}${metric?.higher_is_better === true ? "Больше — лучше" : metric?.higher_is_better === false ? "Меньше — лучше" : "Направление метрики не указано"}</p><div class="metric-divider"></div><dl class="detail-list"><dt>Модель</dt><dd>${esc(b.model?.name || "Не указана")}</dd><dt>Проверочная выборка</dt><dd>${number(split.validation_rows)} строк (${Number.isFinite(split.validation_fraction) ? number(split.validation_fraction * 100) : "—"} %)</dd><dt>Разбиение</dt><dd>${split.stratified === true ? "Со стратификацией" : split.stratified === false ? "Без стратификации" : "Не указано"} · seed ${esc(b.seed ?? "—")}</dd><dt>Обучение и проверка</dt><dd>${duration(timing.validation)}</dd><dt>Полное обучение и прогноз</dt><dd>${duration(timing.refit_and_predict)}</dd></dl><div class="info-box">Локальная проверка на отложенной части train.csv. Это не оценка платформы.</div></div>`;
  const validTitle = !s.exists
    ? "Файл ещё не создан"
    : s.valid === true
      ? "Формат проверен"
      : s.valid === false
        ? "Есть ошибки в CSV"
        : "Проверка недоступна";
  main.innerHTML =
    heading(
      "ИТОГИ И ПРОВЕРКА",
      "Результат исследования",
      "Стартовая метрика и состояние файла, который получит платформа.",
    ) +
    taskStrip(taskInfo()) +
    `<div class="result-grid"><section class="panel metric-card"><div class="panel-head"><div><h2>Стартовая модель</h2><p>Последний отчёт baseline_metrics.json</p></div>${badge("Локальная метрика", "gray")}</div>${baseline}</section><section class="panel"><div class="panel-head"><div><h2>Файл предсказаний</h2><p>Текущее состояние submission.csv</p></div></div><div class="panel-body"><div class="result-state"><span class="result-state-icon ${s.valid === false ? "bad" : s.valid !== true ? "warn" : ""}">${icon(s.valid === true ? "check" : "missing")}</span><div><h3>${validTitle}</h3><p>${s.valid === true ? "Столбцы, значения и порядок ID корректны." : "Результат проверки доступен ниже."}</p></div></div><div class="file-box">${icon("file")}<div><strong>submission.csv</strong><small>${esc(s.path || "Папка выбранной задачи")}</small>${s.exists ? `<small>${Number.isFinite(s.size_bytes) ? number(s.size_bytes / 1024, 1) + " КБ" : "Размер неизвестен"}</small>` : ""}</div></div>${s.exists ? `<a class="button ${s.valid === true ? "primary" : ""}" href="${apiBase()}/submission" download>${icon("download")} Скачать CSV</a>` : '<button class="button" data-action="help">Как получить результат</button>'}${warningList(s.errors)}<p class="run-note">Проверка формата не измеряет качество предсказаний. Текущий CSV мог быть создан более поздним экспериментом, чем отчёт стартовой модели.</p></div></section></div><section class="panel"><div class="panel-head"><h2>История качества</h2>${badge("Нет сопоставимых метрик", "gray")}</div><div class="quality-empty">${icon("chart")}<div><h3>Для сравнения нужны измеренные результаты</h3><p>${esc(data.quality_note || "Журнал агента пока не содержит структурированной истории метрик каждого эксперимента. Сравнивайте модели на одинаковых проверочных строках и сохраняйте метрику вместе с результатом.")}</p></div></div></section>`;
}
function command(text) {
  return `<div class="command-row"><code>${esc(text)}</code><button data-copy="${esc(text)}" aria-label="Скопировать команду" title="Скопировать">${icon("copy")}</button></div>`;
}
function showHelp() {
  const selected = state.task;
  const publicTask = ["regression", "classification"].includes(selected)
    ? selected
    : null;
  const path = publicTask
    ? `data/${publicTask}/task.md`
    : "ПУТЬ_К_ЗАДАЧЕ/task.md";
  $("#help-content").innerHTML =
    `<div class="steps"><div class="help-step"><span>1</span><div class="help-step-content"><h3>Подготовьте данные</h3><p>Откройте PowerShell в корне проекта. Эта команда скачает обе публичные задачи.</p>${command(".\\.venv\\Scripts\\python.exe prepare_data.py all")}</div></div><div class="help-step"><span>2</span><div class="help-step-content"><h3>Получите стартовую модель</h3><p>Обычное обучение без API. Существующий submission.csv сохраняется; для намеренной замены есть параметр --overwrite.</p>${command(`.\\.venv\\Scripts\\python.exe offline_baseline.py "${path}"`)}</div></div><div class="help-step"><span>3</span><div class="help-step-content"><h3>Запустите исследование с Qwen</h3><p>После настройки ключа, адреса и модели в локальном .env. Этот запуск расходует общий API-бюджет команды. После запуска откройте раздел «Запуски».</p>${command(`.\\.venv\\Scripts\\python.exe solution.py "${path}"`)}</div></div></div><p class="help-note">Панель показывает локальные данные и отчёты. Команды выполняются вами в терминале. На macOS/Linux используйте .venv/bin/python вместо Windows-пути. Если задан --data-dir, замените путь data/… в командах на фактическую папку задачи.${publicTask ? "" : " Замените ПУТЬ_К_ЗАДАЧЕ на фактическую папку задачи."}</p>`;
  $("#help-dialog").showModal();
}
let toastTimer;
function toast(text) {
  const node = $("#toast");
  node.textContent = text;
  node.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("visible"), 2500);
}
async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Команда скопирована");
  } catch {
    toast("Не удалось скопировать. Выделите команду вручную.");
  }
}
async function refresh() {
  state.profiles.delete(state.task);
  state.lastRunFingerprint = null;
  await loadTasks();
}
function navigate(view) {
  if (!["data", "runs", "result"].includes(view)) return;
  state.view = view;
  history.replaceState(null, "", `#${view}`);
  renderView();
}
$("#task-select").addEventListener("change", (event) => {
  state.task = event.target.value;
  state.selectedColumn = null;
  state.selectedRun = null;
  state.experiment = null;
  state.runDetail = null;
  renderView();
});
$$("[data-icon]").forEach((node) => (node.innerHTML = icon(node.dataset.icon)));
$$(".nav-item").forEach((button) =>
  button.addEventListener("click", () => navigate(button.dataset.view)),
);
$("#help-button").addEventListener("click", showHelp);
$("#sidebar-help").addEventListener("click", showHelp);
$("#close-help").addEventListener("click", () => $("#help-dialog").close());
$("#help-dialog").addEventListener("click", (event) => {
  if (event.target === $("#help-dialog")) {
    const box = event.target.getBoundingClientRect();
    if (
      event.clientX < box.left ||
      event.clientX > box.right ||
      event.clientY < box.top ||
      event.clientY > box.bottom
    )
      event.target.close();
  }
});
document.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.action === "help") showHelp();
  else if (button.dataset.action === "refresh") refresh();
  else if (button.dataset.column !== undefined)
    loadColumn(button.dataset.column);
  else if (button.dataset.run !== undefined) loadRun(button.dataset.run);
  else if (button.dataset.fileKind) {
    state.fileKind = button.dataset.fileKind;
    loadRunFile();
  } else if (button.dataset.copy !== undefined) copy(button.dataset.copy);
});
window.addEventListener("hashchange", () => navigate(location.hash.slice(1)));
setInterval(async () => {
  if (state.view !== "runs" || !state.task || state.pollBusy || document.hidden)
    return;
  state.pollBusy = true;
  try {
    await loadRuns(state.generation, true);
  } catch {
    const node = $("#last-refresh");
    if (node)
      node.textContent = "Обновление не удалось. Повторим через 4 секунды.";
  } finally {
    state.pollBusy = false;
  }
}, 4000);
if (["data", "runs", "result"].includes(location.hash.slice(1)))
  state.view = location.hash.slice(1);
loadTasks();
