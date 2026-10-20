/* Alfa Lab: local data, controlled experiments and team reports. */
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
  session: null,
  workflow: null,
  runMode: "jobs",
  jobs: [],
  selectedJob: null,
  jobDetail: null,
  jobGeneration: 0,
  jobLogKind: "stdout",
  jobPollBusy: false,
  experimentReports: [],
  publishJob: null,
  deliveryBusy: false,
  launchTask: null,
  launchGeneration: 0,
  workflowUpdated: 0,
  experimentFingerprint: null,
  inspectJob: null,
  currentDatasetFingerprint: null,
};
const main = $("#main");
function apiBase(task = state.task) {
  return `/api/tasks/${encodeURIComponent(task)}`;
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    credentials: "same-origin",
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.method
        ? {
            "Content-Type": "application/json",
            "X-Alfa-Token": state.session?.token || "",
          }
        : {}),
      ...(options.headers || {}),
    },
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
  return `<div class="page-heading"><div><span class="eyebrow">${esc(eyebrow)}</span><h1>${esc(title)}</h1><p class="page-description">${esc(description)}</p></div>${actions ? `<div class="heading-actions"><button class="button" data-action="refresh" title="Обновить данные">${icon("refresh")}<span class="button-label">Обновить</span></button><button class="button" data-action="help" title="Настройка среды">${icon("terminal")}<span class="button-label">Среда</span></button></div>` : ""}</div>`;
}
function taskStrip(profile) {
  const metricName =
    profile?.metric?.name ||
    (state.profile?.id === state.task ? state.profile.metric?.name : null);
  return `<div class="summary-strip">${badge(profile?.task_type === "classification" ? "Классификация" : profile?.task_type === "regression" ? "Регрессия" : "ML-задача", "red")}${badge(profile?.target || taskInfo()?.target || "Таргет не определён")}${metricName && metricName !== "Не определена" ? badge(metricName, "gray") : ""}<span class="subtle-text">${esc(taskInfo()?.name || "")}</span></div>`;
}
function empty(title, message, symbol = "folder", action = true) {
  return `<section class="empty-state"><span class="empty-icon">${icon(symbol)}</span><h2>${esc(title)}</h2><p>${esc(message)}</p>${action ? '<button class="button" data-action="help">Как начать</button>' : ""}</section>`;
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
  return `<div class="bar-chart" role="img" aria-label="${esc(label)}">${bins.map((b) => `<div class="bar-row"><span class="bar-label" title="${esc(b.label)}">${esc(b.label)}</span><div class="bar-track"><div class="bar-fill" style="--width:${(Math.max(0, Number(b.count) || 0) / max) * 100}%"></div></div><span class="bar-value">${number(b.count)}${Number.isFinite(b.percent) ? `<span class="class-percent">${number(b.percent, 1)} %</span>` : ""}</span></div>`).join("")}</div>`;
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
    const [data, session] = await Promise.all([
      api("/api/tasks"),
      api("/api/session"),
    ]);
    if (generation === state.generation) state.session = session;
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
          "Загрузите две публичные задачи, чтобы начать обучение.",
          "folder",
          false,
        ) +
        '<div class="empty-start action-buttons"><button class="button primary" data-launch="prepare">Загрузить публичные данные</button><span class="subtle-text">Без API · потребуется интернет</span></div>';
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
  state.jobGeneration++;
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
    if (state.view === "runs") await loadJobsPage(generation);
    else await loadTasks();
    return;
  }
  main.innerHTML = loading(
    state.view === "data"
      ? "Считаем статистику по таблицам…"
      : "Читаем локальные отчёты…",
  );
  try {
    const workflow = await api(`${apiBase()}/workflow`);
    if (generation !== state.generation) return;
    state.workflow = workflow;
    if (state.view === "data" && !workflow.task?.available) {
      main.innerHTML =
        heading(
          "ПОДГОТОВКА",
          "Данные ещё не готовы",
          workflow.task?.error || "Загрузите файлы задачи.",
        ) +
        '<section class="empty-state"><span class="empty-icon">' +
        icon("folder") +
        '</span><h2>Подготовьте публичные задачи</h2><p>Файлы загрузятся в локальную папку данных.</p><button class="button primary" data-launch="prepare">Загрузить данные</button></section>';
      mountWorkflow();
      return;
    }
    if (state.view === "data") await loadData(generation);
    else if (state.view === "runs") await loadRuns(generation);
    else await loadResult(generation);
    if (generation === state.generation) mountWorkflow();
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
      "Проверьте данные и обучите первую модель.",
    ) +
    taskStrip(profile) +
    actionPanel() +
    `<div class="stats-grid">${stat("Обучающая выборка", number(profile.train_rows), "строк в train.csv", "rows")}${stat("Тестовая выборка", number(profile.test_rows), "строк в test.csv", "test")}${stat("Входные признаки", number(profile.feature_count), "без идентификатора и таргета", "layers")}${stat("Пропуски в train", percent(profile.missing_percent), `${number(profile.missing_cells)} пустых ячеек`, "missing")}</div>` +
    warningList(profile.warnings) +
    `<div class="chart-grid"><section class="panel target-classes"><div class="panel-head"><div><h2>Распределение таргета</h2><p>${esc(profile.target)} · обучающая выборка</p></div><span class="panel-label"><span class="mini-dot"></span> TRAIN</span></div><div class="panel-body">${chart(profile.target_distribution, "Распределение таргета " + profile.target)}${profile.target_distribution?.missing_count ? `<p class="chart-footnote">Пропущено: ${number(profile.target_distribution.missing_count)}</p>` : ""}</div></section>${qualityPanel(profile)}</div>` +
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
  queued: ["В очереди", "gray"],
  cancelled: ["Остановлен", "gray"],
  interrupted: ["Прерван", "amber"],
};
function statusBadge(status) {
  const v = STATUS[status] || ["Неизвестный статус", "gray"];
  return badge(v[0], v[1]);
}
async function loadLegacyRuns(generation, quiet = false) {
  const task = state.task;
  const data = await api(legacyEndpoint());
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
    const detail = await api(legacyEndpoint(id));
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
    const data = await api(legacyEndpoint(run, file.name));
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
async function loadLegacyResult(generation) {
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
      "Сравните эксперименты и выберите CSV для платформы.",
    ) +
    taskStrip(taskInfo()) +
    `<div class="result-grid"><section class="panel metric-card"><div class="panel-head"><div><h2>Стартовая модель</h2><p>Последний отчёт baseline_metrics.json</p></div>${badge("Локальная метрика", "gray")}</div>${baseline}</section><section class="panel"><div class="panel-head"><div><h2>Файл предсказаний</h2><p>Текущее состояние submission.csv</p></div></div><div class="panel-body"><div class="result-state"><span class="result-state-icon ${s.valid === false ? "bad" : s.valid !== true ? "warn" : ""}">${icon(s.valid === true ? "check" : "missing")}</span><div><h3>${validTitle}</h3><p>${s.valid === true ? "Столбцы, значения и порядок ID корректны." : "Результат проверки доступен ниже."}</p></div></div><div class="file-box">${icon("file")}<div><strong>submission.csv</strong><small>${esc(s.path || "Папка выбранной задачи")}</small>${s.exists ? `<small>${Number.isFinite(s.size_bytes) ? number(s.size_bytes / 1024, 1) + " КБ" : "Размер неизвестен"}</small>` : ""}</div></div>${s.exists ? `<a class="button ${s.valid === true ? "primary" : ""}" href="${apiBase()}/submission" download>${icon("download")} Скачать CSV</a>` : '<button class="button" data-action="help">Как получить результат</button>'}${warningList(s.errors)}<p class="run-note">Проверка формата не измеряет качество предсказаний. Текущий CSV мог быть создан более поздним экспериментом, чем отчёт стартовой модели.</p></div></section></div><section class="panel"><div class="panel-head"><h2>История качества</h2>${badge("Нет сопоставимых метрик", "gray")}</div><div class="quality-empty">${icon("chart")}<div><h3>Для сравнения нужны измеренные результаты</h3><p>${esc(data.quality_note || "Журнал агента пока не содержит структурированной истории метрик каждого эксперимента. Сравнивайте модели на одинаковых проверочных строках и сохраняйте метрику вместе с результатом.")}</p></div></div></section>`;
}
function command(text) {
  return `<div class="command-row"><code>${esc(text)}</code><button data-copy="${esc(text)}" aria-label="Скопировать команду" title="Скопировать">${icon("copy")}</button></div>`;
}
function showHelp() {
  const apiInfo = state.session?.api || {};
  $("#help-title").textContent = "Настройка среды";
  $("#help-content").innerHTML =
    `<div class="steps"><div class="help-step"><span>1</span><div class="help-step-content"><h3>Запуски на этом компьютере</h3><p>Baseline и ExtraTrees можно обучать кнопками в панели. Результаты каждого эксперимента сохраняются отдельно.</p></div></div><div class="help-step"><span>2</span><div class="help-step-content"><h3>Подключение Qwen</h3><p>${apiInfo.ready ? "Настройки API заполнены. Подключение проверяется при запуске исследования." : "Заполните настройки организаторов в локальном .env и перезапустите сервер."}</p>${!apiInfo.ready ? `<div class="command-row"><code>${esc((apiInfo.missing || ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]).join("\n"))}</code></div>` : ""}</div></div><div class="help-step"><span>3</span><div class="help-step-content"><h3>Работа команды</h3><p>В разделе «Результат» можно скачать отчёт JSON или ZIP-пакет запуска и добавить его в панель коллеги. Импорт добавляет только отчёт; текущий CSV выбирается отдельно.</p></div></div></div><p class="help-note">Каждый участник запускает панель на своём компьютере. Общего онлайн-сервера в этой версии нет.</p>`;
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
  state.inspectJob = null;
  state.selectedJob = null;
  state.jobDetail = null;
  state.workflow = null;
  state.experimentReports = [];
  state.experimentFingerprint = null;
  state.currentDatasetFingerprint = null;
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
  if (button.dataset.closeDialog) $("#" + button.dataset.closeDialog)?.close();
  else if (button.dataset.launch) showLaunch(button.dataset.launch);
  else if (button.dataset.openJob)
    openJob(button.dataset.openJob, button.dataset.taskId || null);
  else if (button.dataset.job) selectJob(button.dataset.job);
  else if (button.dataset.jobLog) {
    state.jobLogKind = button.dataset.jobLog;
    renderJobLog();
  } else if (button.dataset.runMode) {
    state.runMode = button.dataset.runMode;
    if (state.runMode === "jobs") state.inspectJob = null;
    renderView();
  } else if (button.dataset.workflowView) navigate(button.dataset.workflowView);
  else if (button.dataset.action === "inspect-agent") {
    state.inspectJob = state.selectedJob;
    state.runMode = "agent";
    state.selectedRun = null;
    renderView();
  } else if (button.dataset.action === "stop-job") stopJob();
  else if (button.dataset.action === "publish-job")
    showPublish(button.dataset.publishId || state.selectedJob);
  else if (button.dataset.action === "confirm-publish") publishJob();
  else if (button.dataset.action === "build-delivery") downloadDelivery(button);
  else if (button.dataset.action === "import-reports")
    $("#report-file").click();
  else if (button.dataset.action === "open-data") {
    state.view = "data";
    loadTasks();
  } else if (button.dataset.action === "help") showHelp();
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
setInterval(pollWorkspace, 1500);
if (["data", "runs", "result"].includes(location.hash.slice(1)))
  state.view = location.hash.slice(1);
loadTasks();

// Controlled jobs and team reports. The server owns commands and file access.
const ACTION_NAMES = {
  baseline: "Baseline",
  extra_trees: "ExtraTrees",
  agent: "Qwen",
  prepare: "Загрузка данных",
};
const STAGE_NAMES = {
  preparing: "Подготовка данных",
  training: "Обучение модели",
  validating: "Проверка на holdout",
  refitting: "Обучение на всей выборке",
  finished: "Работа завершена",
  research: "Исследование с Qwen",
  downloading: "Загрузка публичных данных",
};
const runningJob = (job) => job && ["queued", "running"].includes(job.status);
const post = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify(body) });
const jobPath = (id, task = state.task) =>
  `/api/jobs/${encodeURIComponent(id)}${task ? "?task_id=" + encodeURIComponent(task) : ""}`;
function workflowStrip() {
  const w = state.workflow || {},
    globalActive = state.workflow
      ? state.workflow.active_job
      : state.session?.active_job,
    active = globalActive?.task_id === state.task ? globalActive : null;
  const currentJob =
    w.latest_job?.dataset_fingerprint === w.dataset_fingerprint
      ? w.latest_job
      : null;
  const model = Boolean(w.baseline?.exists || currentJob?.metric);
  const localModel = currentJob?.status === "completed" && currentJob?.metric;
  const steps = [
    {
      name: "Данные",
      detail: w.task?.available ? "Готовы к работе" : "Нужно загрузить",
      done: !!w.task?.available,
      view: "data",
    },
    {
      name: "Модель",
      detail: localModel
        ? "Есть измеренный запуск"
        : w.baseline?.exists
          ? "Найден прежний baseline"
          : "Обучите baseline",
      done: model,
      active: runningJob(active) && active.action === "baseline",
      view: "runs",
    },
    {
      name: "Сравнение",
      detail:
        runningJob(active) && active.action !== "baseline"
          ? "Выполняется"
          : w.experiments_count
            ? `${number(w.experiments_count)} экспериментов`
            : "Сравните другую модель",
      done: false,
      active: runningJob(active) && active.action !== "baseline",
      view: "runs",
    },
    {
      name: "Результат",
      detail: w.published_job_id
        ? "Запуск выбран для CSV"
        : w.submission?.valid
          ? "Найден прежний CSV"
          : "Выберите CSV запуска",
      done: Boolean(w.published_job_id),
      view: "result",
    },
  ];
  return `<div class="workflow-strip" id="workflow-strip" aria-label="Этапы работы">${steps.map((step, i) => `<button class="workflow-step ${step.done ? "done" : ""} ${step.active ? "active" : ""}" data-workflow-view="${step.view}"><span class="workflow-number">${step.done ? icon("check") : i + 1}</span><span><strong>${step.name}</strong><small>${esc(step.detail)}</small></span></button>`).join("")}</div>`;
}
function mountWorkflow() {
  if (!state.task) return;
  const current = $("#workflow-strip");
  if (current) current.outerHTML = workflowStrip();
  else {
    const headingNode = $(".page-heading");
    if (headingNode)
      headingNode.insertAdjacentHTML("afterend", workflowStrip());
  }
}
function actionPanel() {
  const active = state.workflow
    ? state.workflow.active_job
    : state.session?.active_job;
  const ready = state.workflow?.api?.ready ?? state.session?.api?.ready;
  if (runningJob(active))
    return `<section class="next-action"><div><h2>${esc(ACTION_NAMES[active.action] || "Эксперимент")} выполняется</h2><p>${esc(STAGE_NAMES[active.stage] || "Работа продолжается")}</p></div><button class="button primary" data-open-job="${esc(active.id)}" data-task-id="${esc(active.task_id || "")}">Открыть запуск ${icon("activity")}</button></section>`;
  const latest = state.workflow?.latest_job;
  if (
    latest?.action === "agent" &&
    latest.dataset_fingerprint === state.workflow?.dataset_fingerprint &&
    !Number.isFinite(latest.metric?.value) &&
    ["completed", "failed", "cancelled", "interrupted"].includes(latest.status)
  ) {
    const title =
      latest.agent_status === "fallback"
        ? "Qwen сохранил резервный CSV"
        : latest.status === "completed"
          ? "Исследование Qwen завершено"
          : latest.status === "failed"
            ? "Исследование Qwen завершилось с ошибкой"
            : "Исследование Qwen остановлено";
    return `<section class="next-action"><div><h2>${esc(title)}</h2><p>Сопоставимая метрика не записана. Проверьте журнал и код исследования.</p></div><div class="action-buttons"><button class="button primary" data-open-job="${esc(latest.id)}" data-task-id="${esc(latest.task_id || "")}">Открыть исследование ${icon("activity")}</button><button class="button" data-workflow-view="result">Все результаты</button></div></section>`;
  }
  const measured =
    latest?.status === "completed" &&
    latest.dataset_fingerprint === state.workflow?.dataset_fingerprint &&
    Number.isFinite(latest.metric?.value);
  const firstComparison = measured && latest.action === "baseline";
  const title = measured
    ? firstComparison
      ? "Стартовая точка готова"
      : "Эксперимент завершён — сравните результат"
    : state.workflow?.baseline?.exists
      ? "Есть прежний baseline. Сохраните первый запуск"
      : "Начните со стартовой модели";
  const description = measured
    ? `${latest.metric.name}: ${number(latest.metric.value, 6)} · ${latest.metric.higher_is_better ? "больше лучше" : "меньше лучше"}`
    : "Baseline обучится без API и даст точку сравнения.";
  const primary = measured
    ? firstComparison
      ? '<button class="button primary" data-launch="extra_trees">Сравнить с ExtraTrees</button>'
      : '<button class="button primary" data-workflow-view="result">Сравнить результаты</button>'
    : `<button class="button primary" data-launch="baseline">${icon("test")} Обучить baseline</button>`;
  return `<section class="next-action"><div><h2>${esc(title)}</h2><p>${esc(description)}</p></div><div class="action-buttons">${primary}${measured ? '<button class="button" data-launch="baseline">Повторить baseline</button>' : '<button class="button" data-launch="extra_trees">Сравнить модели</button>'}<button class="button" data-launch="agent">${ready ? "Исследовать с Qwen" : "Подключить Qwen"} ${icon("activity")}</button></div></section>`;
}
function qualityPanel(profile) {
  const q = profile.quality || {},
    changes = q.missing_changes || [],
    constants = q.constant_features || [],
    categories = q.new_categories || [],
    shifts = q.numeric_shift || [];
  const duplicateIds =
    Number.isFinite(q.train_duplicate_ids) &&
    Number.isFinite(q.test_duplicate_ids)
      ? q.train_duplicate_ids + q.test_duplicate_ids
      : null;
  const testMissing = Number.isFinite(profile.test_missing_percent)
    ? percent(profile.test_missing_percent)
    : null;
  const detailLines = [
    q.id_overlap_count != null
      ? `ID в обеих выборках: ${number(q.id_overlap_count)}.`
      : "",
    q.train_duplicate_feature_rows != null
      ? `Повторы строк признаков: train ${number(q.train_duplicate_feature_rows)}, test ${number(q.test_duplicate_feature_rows)}.`
      : "",
    constants.length
      ? `Постоянные признаки: ${constants.slice(0, 12).join(", ")}${constants.length > 12 ? "…" : ""}.`
      : "",
    categories.length
      ? `Новые категории в test: ${categories
          .map((c) => `${c.name} (${c.count})`)
          .slice(0, 8)
          .join(", ")}.`
      : "",
    shifts.length
      ? `Изменение числовых распределений: ${shifts
          .slice(0, 5)
          .map(
            (c) =>
              `${c.name}: медиана ${number(c.train_median, 3)} → ${number(c.test_median, 3)}`,
          )
          .join("; ")}.`
      : "",
  ].filter(Boolean);
  return `<section class="panel quality-summary"><div class="panel-head"><div><h2>Проверка train / test</h2><p>Короткая диагностика перед обучением</p></div>${badge("Данные", "gray")}</div><div class="panel-body"><div class="quality-grid"><div class="quality-item"><strong>${percent(profile.missing_percent)}</strong><span>Пропуски в train${testMissing ? ` · test ${testMissing}` : ""}</span></div><div class="quality-item"><strong>${number(duplicateIds)}</strong><span>Повторяющихся ID в выборках</span></div><div class="quality-item"><strong>${number(constants.length)}</strong><span>Постоянных признаков</span></div><div class="quality-item"><strong>${number(categories.length)}</strong><span>Столбцов с новыми категориями</span></div></div>${q.class_balance?.ratio ? `<p class="chart-footnote">Самый частый класс встречается в ${number(q.class_balance.ratio, 1)} раза чаще самого редкого.</p>` : ""}<details class="quality-details"><summary>Сравнение train/test</summary><div class="quality-details-content">${detailLines.map((text) => `<p>${esc(text)}</p>`).join("")}${
    changes.length
      ? `<p>Доля пропусков по столбцам:</p><ul>${changes
          .slice(0, 8)
          .map(
            (c) =>
              `<li>${esc(c.name)}: ${percent(c.train_percent)} → ${percent(c.test_percent)}</li>`,
          )
          .join("")}</ul>`
      : ""
  }${!detailLines.length && !changes.length ? "<p>Дополнительных сигналов нет.</p>" : ""}<p>Различия между выборками требуют проверки и не всегда означают ошибку.</p></div></details>${profile.missing_cells > 0 ? `<details class="quality-details"><summary>Где есть пропуски</summary>${missingChart(profile)}</details>` : ""}</div></section>`;
}
function runTabs() {
  return `<div class="view-tabs" id="run-view-tabs" role="tablist" aria-label="Источник запусков"><button class="view-tab ${state.runMode === "jobs" ? "active" : ""}" data-run-mode="jobs" role="tab" aria-selected="${state.runMode === "jobs"}">Запуски из панели</button><button class="view-tab ${state.runMode === "agent" ? "active" : ""}" data-run-mode="agent" role="tab" aria-selected="${state.runMode === "agent"}">${state.inspectJob ? "Код Qwen" : "Из терминала"}</button></div>`;
}
async function loadRuns(generation, quiet = false) {
  if (state.runMode === "agent") {
    await loadLegacyRuns(generation, quiet);
    if (generation === state.generation && !$("#run-view-tabs"))
      $(".summary-strip")?.insertAdjacentHTML("afterend", runTabs());
  } else await loadJobsPage(generation, quiet);
}
async function loadJobsPage(generation, quiet = false) {
  const task = state.task;
  const data = await api(
    `/api/jobs${task ? "?task_id=" + encodeURIComponent(task) : ""}`,
  );
  if (generation !== state.generation) return;
  state.jobs = Array.isArray(data.jobs)
    ? data.jobs.filter((job) =>
        task ? job.task_id === task : job.task_id == null,
      )
    : [];
  if (!state.jobs.some((job) => job.id === state.selectedJob))
    state.selectedJob =
      state.jobs.find(runningJob)?.id || state.jobs[0]?.id || null;
  if (!quiet || !$("#job-list")) {
    main.innerHTML =
      heading(
        "ЭКСПЕРИМЕНТЫ",
        "Запуски",
        "Запустите модель и наблюдайте за работой.",
      ) +
      (state.task ? taskStrip(taskInfo()) : "") +
      runTabs() +
      `<div class="run-start-actions"><button class="button primary" data-launch="baseline" ${state.task ? "" : "disabled"}>Обучить baseline</button><button class="button" data-launch="extra_trees" ${state.task ? "" : "disabled"}>Сравнить модели</button><span class="subtle-text">На этом компьютере</span></div><div class="jobs-layout" style="margin-top:20px"><section class="panel run-list-panel"><div class="panel-head"><h2>История</h2><span class="subtle-text" id="job-count"></span></div><div id="job-list" class="job-list"></div></section><section id="job-detail"></section></div>`;
    mountWorkflow();
  }
  renderJobList();
  if (state.selectedJob) await loadJobDetail(state.selectedJob, quiet);
  else
    $("#job-detail").innerHTML =
      `<section class="panel"><div class="job-empty"><p>Здесь будут только запуски из этой панели. Начните с baseline, чтобы записать первую метрику.</p><button class="button primary" data-launch="${state.task ? "baseline" : "prepare"}">${state.task ? "Обучить первую модель" : "Загрузить данные"}</button></div></section>`;
}
function renderJobList() {
  if (!$("#job-list")) return;
  $("#job-count").textContent = String(state.jobs.length);
  $("#job-list").innerHTML =
    state.jobs
      .map(
        (job) =>
          `<button class="job-item ${job.id === state.selectedJob ? "selected" : ""}" data-job="${esc(job.id)}" aria-pressed="${job.id === state.selectedJob}"><h3>${esc(ACTION_NAMES[job.action] || job.model || "Эксперимент")}</h3>${statusBadge(job.agent_status === "fallback" ? "fallback" : job.status)}<p>${esc(job.author || "Без автора")} · ${duration(job.elapsed_seconds)}</p>${["baseline", "extra_trees"].includes(job.action) ? `<p class="job-validation-label">${esc(validationLabel(job))}</p>` : ""}<p>${esc(date(job.started_at))}</p></button>`,
      )
      .join("") || '<p class="subtle-text">Запусков пока нет</p>';
}
function currentLocalResult(job) {
  const fingerprint =
    state.workflow?.dataset_fingerprint || state.currentDatasetFingerprint;
  return (
    job?.source === "local" &&
    job?.task_id === state.task &&
    job?.status === "completed" &&
    fingerprint &&
    job.dataset_fingerprint === fingerprint
  );
}
function jobBundleLink(job, label = "Скачать пакет запуска", small = false) {
  if (!currentLocalResult(job) || job.output_valid !== true) return "";
  return `<a class="button${small ? " small" : ""}" href="/api/jobs/${encodeURIComponent(job.id)}/bundle?task_id=${encodeURIComponent(state.task)}" download>${icon("download")} ${esc(label)}</a>`;
}
function validationStrategy(job) {
  const strategy = job?.split?.strategy || job?.validation?.strategy;
  return ["random", "time", "grouped"].includes(strategy) ? strategy : null;
}
function validationLabel(job) {
  const strategy = validationStrategy(job);
  if (strategy)
    return {
      random: "Случайные 80/20",
      time: "По последнему периоду",
      grouped: "По группам дублей",
    }[strategy];
  return job?.source === "local" &&
    ["baseline", "extra_trees"].includes(job?.action) &&
    job?.split
    ? "Прежняя проверка 80/20"
    : "Схема не указана";
}
function validationEvidence(job) {
  const split = job?.split || {},
    strategy = validationStrategy(job);
  if (strategy === "time") {
    const lower = split.train_period_min,
      upper = split.train_period_max,
      latest = split.validation_period;
    const periods =
      lower != null && upper != null && latest != null
        ? `Обучение ${lower === upper ? lower : `${lower}–${upper}`} → проверка ${latest}`
        : "Ранние периоды → последний";
    return `${split.time_column || job?.validation?.time_column || "Период"} · ${periods}`;
  }
  if (strategy === "grouped") {
    const counts =
      Number.isFinite(split.train_groups) &&
      Number.isFinite(split.validation_groups)
        ? `Группы ${number(split.train_groups)} / ${number(split.validation_groups)}`
        : "Одинаковые строки в одной части";
    return `${counts}${Number.isFinite(split.group_overlap) ? ` · пересечений ${number(split.group_overlap)}` : ""}`;
  }
  return [
    split.stratified === true ? "Со стратификацией" : "",
    Number.isFinite(split.seed) ? `seed ${number(split.seed)}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
}
function validationView(job, compact = false) {
  const evidence = validationEvidence(job),
    fraction = job?.split?.validation_fraction;
  return `<div class="validation-summary${compact ? " compact" : ""}"><span class="validation-tag validation-${validationStrategy(job) || "legacy"}">${esc(validationLabel(job))}</span>${evidence ? `<span class="validation-evidence">${esc(evidence)}</span>` : ""}${compact && Number.isFinite(fraction) ? `<span class="validation-evidence">Проверка ${number(fraction * 100, 1)} % строк</span>` : ""}</div>`;
}
function measuredComparison(job) {
  if (
    !currentLocalResult(job) ||
    !job.comparison_key ||
    !Number.isFinite(job.metric?.value) ||
    typeof job.metric.higher_is_better !== "boolean"
  )
    return null;
  const earlier = state.jobs.filter(
    (row) =>
      row.id !== job.id &&
      currentLocalResult(row) &&
      row.comparison_key === job.comparison_key &&
      row.dataset_fingerprint === job.dataset_fingerprint &&
      row.metric?.name === job.metric.name &&
      row.metric?.higher_is_better === job.metric.higher_is_better &&
      Number.isFinite(row.metric?.value) &&
      row.started_at <= job.started_at,
  );
  const reference =
    earlier.find((row) => row.action === "baseline") ||
    earlier.reduce(
      (best, row) =>
        !best ||
        (job.metric.higher_is_better
          ? row.metric.value > best.metric.value
          : row.metric.value < best.metric.value)
          ? row
          : best,
      null,
    );
  if (!reference)
    return {
      title:
        job.action === "baseline" ? "Стартовая точка" : "Первая оценка модели",
      note: "Первая оценка на этой схеме проверки.",
      tone: "neutral",
      symbol: "test",
    };
  const delta = job.metric.value - reference.metric.value;
  const improvement = job.metric.higher_is_better ? delta : -delta;
  const equal = Math.abs(delta) < 1e-9;
  const referenceName =
    reference.action === "baseline" ? "baseline" : "лучшего прежнего запуска";
  return {
    title: equal
      ? `На уровне ${referenceName}`
      : `${improvement > 0 ? "Лучше" : "Хуже"} ${referenceName}`,
    note: equal
      ? "Метрика повторилась на тех же проверочных строках."
      : `${delta > 0 ? "+" : "−"}${number(Math.abs(delta), 6)} ${job.metric.name} · те же проверочные строки`,
    tone: equal ? "neutral" : improvement > 0 ? "better" : "worse",
    symbol: equal ? "check" : improvement > 0 ? "arrowup" : "arrowdown",
  };
}
function jobMetricView(job) {
  if (!Number.isFinite(job.metric?.value)) return "";
  const outcome = measuredComparison(job);
  const next = currentLocalResult(job)
    ? job.action === "baseline"
      ? '<button class="button small" data-launch="extra_trees">Следующий шаг: сравнить с ExtraTrees</button>'
      : '<button class="button small" data-workflow-view="result">Сравнить все результаты</button>'
    : "";
  return `<section class="job-metric">${outcome ? `<div class="job-outcome ${outcome.tone}"><span class="job-outcome-icon">${icon(outcome.symbol)}</span><div><h3>${esc(outcome.title)}</h3><p>${esc(outcome.note)}</p></div></div>` : ""}<div class="job-metric-value"><strong>${number(job.metric.value, 6)}</strong><span>${esc(job.metric.name)}</span></div><p>${esc(job.model || ACTION_NAMES[job.action])} · проверка${job.split?.validation_fraction != null ? " на " + number(job.split.validation_fraction * 100, 1) + " % данных" : " на отложенных данных"} · ${job.metric.higher_is_better ? "больше лучше" : "меньше лучше"}</p>${validationView(job)}${next ? `<div class="job-next-step">${next}</div>` : ""}</section>`;
}
async function selectJob(id) {
  state.selectedJob = id;
  state.jobDetail = null;
  state.jobLogKind = "stdout";
  renderJobList();
  if ($("#job-detail")) $("#job-detail").innerHTML = inlineLoading();
  await loadJobDetail(id);
}
async function loadJobDetail(id, quiet = false) {
  const task = state.task,
    viewGeneration = state.generation,
    jobGeneration = ++state.jobGeneration;
  try {
    const detail = await api(jobPath(id, task));
    if (
      viewGeneration !== state.generation ||
      jobGeneration !== state.jobGeneration ||
      id !== state.selectedJob
    )
      return;
    if (
      quiet &&
      $("#job-log-output") &&
      JSON.stringify(detail) === JSON.stringify(state.jobDetail)
    ) {
      const stamp = $("#job-refresh-time");
      if (stamp)
        stamp.textContent = `Проверено в ${new Date().toLocaleTimeString("ru-RU")}`;
      return;
    }
    const focusedAction = quiet && document.activeElement?.dataset?.action;
    state.jobDetail = detail;
    const job = detail.job || {},
      scroll = quiet
        ? {
            top: $("#job-log-output")?.scrollTop || 0,
            left: $("#job-log-output")?.scrollLeft || 0,
          }
        : null;
    const validMetric = job.metric && Number.isFinite(job.metric.value);
    const stopped = ["cancelled", "interrupted"].includes(job.status);
    $("#job-detail").innerHTML =
      `<div class="run-detail-header"><div><h2>${esc(ACTION_NAMES[job.action] || "Эксперимент")}</h2><p>${esc(date(job.started_at))} · ${esc(job.author || "Без автора")}</p></div>${statusBadge(job.agent_status === "fallback" ? "fallback" : job.status)}</div>${job.hypothesis ? `<p class="job-hypothesis">${esc(job.hypothesis)}</p>` : ""}<div class="job-stage">${runningJob(job) ? '<span class="spinner"></span>' : icon(job.status === "completed" ? "check" : stopped ? "clock" : "missing")}<span>${esc(job.status === "cancelled" ? "Остановлен пользователем" : job.status === "failed" ? "Запуск завершился с ошибкой" : STAGE_NAMES[job.stage] || "Подготовка")}</span><span style="margin-left:auto">${duration(job.elapsed_seconds)}</span></div>${job.error ? `<div class="warning-box">${esc(job.error)}</div>` : ""}${job.agent_status === "fallback" ? `<div class="warning-box">Резервный CSV · ${esc({ api_error: "ошибка обращения к API", time_limit: "достигнут лимит времени", step_limit: "достигнут лимит шагов", truncated_response: "ответ модели обрезан" }[job.agent_reason] || job.agent_reason || "исследование не завершилось штатно")}.</div>` : ""}${validMetric ? jobMetricView(job) : job.status === "completed" && job.action === "agent" ? `<div class="info-box">${job.agent_status === "fallback" ? "Сохранён резервный CSV. Исследование не завершилось штатно." : "Процесс Qwen завершён."} Качество не измерено в сопоставимом формате. Откройте код агента и проверьте его эксперимент.</div>` : ""}<div class="job-actions">${runningJob(job) ? `<button class="button danger" data-action="stop-job">Остановить</button>` : ""}${job.status === "completed" && job.output_valid === true ? `<a class="button" href="/api/jobs/${encodeURIComponent(job.id)}/submission${task ? "?task_id=" + encodeURIComponent(task) : ""}" download>${icon("download")} Скачать CSV</a>${job.published ? badge("CSV используется", "green") : `<button class="button primary" data-action="publish-job" data-publish-id="${esc(job.id)}">Использовать CSV</button>`}` : ""}${jobBundleLink(job)}${job.action === "prepare" && job.status === "completed" ? '<button class="button primary" data-action="open-data">Открыть данные</button>' : ""}${job.action === "agent" ? '<button class="button" data-action="inspect-agent">Код агента</button>' : ""}<span class="subtle-text">${job.action === "agent" ? "Qwen · общий API-бюджет" : "Без API"}</span></div>${job.api ? `<div class="usage-row">${usageView(job.api)}</div>` : ""}<section class="panel job-logs"><div class="panel-head"><h2>Журнал запуска</h2><span class="job-log-time" id="job-refresh-time">Обновляется автоматически</span></div><div class="code-tabs" role="tablist" aria-label="Вывод процесса"><button class="code-tab ${state.jobLogKind === "stdout" ? "active" : ""}" data-job-log="stdout" role="tab" aria-selected="${state.jobLogKind === "stdout"}">Вывод</button><button class="code-tab ${state.jobLogKind === "stderr" ? "active" : ""}" data-job-log="stderr" role="tab" aria-selected="${state.jobLogKind === "stderr"}">Ошибки</button></div><pre id="job-log-output" class="code-content" role="tabpanel"></pre><div class="code-footnote">${detail.truncated ? "Показан ограниченный фрагмент журнала" : "Вывод процесса"} · ${esc(job.id?.slice(0, 8) || "")}</div></section>`;
    renderJobLog();
    if (scroll) {
      $("#job-log-output").scrollTop = scroll.top;
      $("#job-log-output").scrollLeft = scroll.left;
    }
    if (focusedAction === "stop-job") {
      $('[data-action="stop-job"]')?.focus({ preventScroll: true });
    }
  } catch (error) {
    if (
      viewGeneration === state.generation &&
      jobGeneration === state.jobGeneration &&
      $("#job-detail")
    )
      $("#job-detail").innerHTML = errorView(
        error,
        "Не удалось прочитать запуск",
      );
  }
}
function renderJobLog() {
  const output = $("#job-log-output");
  if (!output) return;
  const text = state.jobDetail?.[state.jobLogKind];
  output.textContent =
    typeof text === "string" && text
      ? text
      : state.jobLogKind === "stderr"
        ? "Вывод ошибок пуст."
        : "Вывод ещё не появился.";
  $$("[data-job-log]").forEach((button) => {
    const selected = button.dataset.jobLog === state.jobLogKind;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
}
function launchValidationConfig(action, profile) {
  const numeric = (profile?.columns || []).filter(
    (column) => column.role === "feature" && column.dtype === "numeric",
  );
  const fallback = {
    strategy: "random",
    time_column: numeric.find((column) => column.name === "Year_Factor")?.name || "",
  };
  if (action !== "extra_trees") return fallback;
  const eligible = (job) =>
    currentLocalResult(job) &&
    ["baseline", "extra_trees"].includes(job.action);
  const selected =
    state.jobDetail?.job?.id === state.selectedJob
      ? state.jobDetail.job
      : state.jobs.find((job) => job.id === state.selectedJob);
  const reference = eligible(selected)
    ? selected
    : [...state.jobs, ...state.experimentReports]
        .filter(eligible)
        .sort((a, b) => String(b.started_at).localeCompare(String(a.started_at)))[0];
  if (!reference) return fallback;
  const strategy = validationStrategy(reference) || "random",
    timeColumn = reference.split?.time_column || reference.validation?.time_column;
  if (
    strategy === "time" &&
    profile.task_type === "regression" &&
    numeric.some((column) => column.name === timeColumn)
  )
    return { strategy, time_column: timeColumn };
  if (strategy === "grouped" && profile.task_type === "classification")
    return { ...fallback, strategy };
  return fallback;
}
function validationFields(profile, selected) {
  const numeric = (profile?.columns || []).filter(
    (column) => column.role === "feature" && column.dtype === "numeric",
  );
  const options = [["random", "Случайные 80/20", "С фиксированным seed 42", true]];
  if (profile?.task_type === "regression")
    options.push(["time", "Последний период", "Ранние → последний", numeric.length > 0]);
  if (profile?.task_type === "classification")
    options.push(["grouped", "Разделить дубли", "Копии только в одной части", true]);
  return `<fieldset id="validation-controls" class="validation-controls"><legend>Проверка качества</legend><div class="validation-options">${options.map(([value, label, description, enabled]) => `<label class="validation-option"><input type="radio" name="validation_strategy" value="${value}" ${selected.strategy === value ? "checked" : ""} ${enabled ? "" : "disabled"}><span><strong>${label}</strong><small>${enabled ? description : "Нет числовых признаков"}</small></span></label>`).join("")}</div><label class="form-field validation-time-column" id="validation-time-field" hidden>Столбец периода<select name="time_column" required disabled><option value="">Выберите столбец периода</option>${numeric.map((column) => `<option value="${esc(column.name)}" ${selected.time_column === column.name ? "selected" : ""}>${esc(column.name)}</option>`).join("")}</select></label></fieldset>`;
}
function updateLaunchFields() {
  const form = $("#launch-form"),
    value = $('input[name="recipe"]:checked', form)?.value,
    offline = ["baseline", "extra_trees"].includes(value),
    strategy = $('input[name="validation_strategy"]:checked', form)?.value;
  $("#launch-cost").textContent =
    value === "agent"
      ? "Qwen · расходует общий API-бюджет"
      : "Без API · обучение на этом компьютере";
  $("#launch-cost").classList.toggle("paid", value === "agent");
  $("#qwen-time-field").hidden = value !== "agent";
  $("#hypothesis-label").textContent =
    value === "agent" ? "Гипотеза для Qwen" : "Заметка к эксперименту";
  form.elements.minutes.disabled = value !== "agent";
  $("#validation-controls").hidden = !offline;
  $("#validation-controls").disabled = !offline;
  $("#validation-time-field").hidden = !offline || strategy !== "time";
  form.elements.time_column.disabled = !offline || strategy !== "time";
}
async function showLaunch(action = "baseline") {
  if (action === "prepare") {
    startPrepare();
    return;
  }
  if (!state.task) return;
  const task = state.task,
    generation = ++state.launchGeneration,
    viewGeneration = state.generation;
  state.launchTask = task;
  $("#launch-title").textContent =
    action === "agent"
      ? "Исследование с Qwen"
      : action === "baseline"
        ? "Измерить стартовое качество"
        : "Сравнить другую модель";
  const ready = state.workflow?.api?.ready ?? state.session?.api?.ready;
  if (action === "agent" && !ready) {
    showHelp();
    return;
  }
  let profile = state.profiles.get(task);
  const fingerprint = state.workflow?.dataset_fingerprint || state.currentDatasetFingerprint;
  try {
    if (!profile || (fingerprint && profile.dataset_fingerprint !== fingerprint))
      profile = await api(`${apiBase(task)}/profile`);
  } catch (error) {
    if (state.task === task && generation === state.launchGeneration && viewGeneration === state.generation)
      toast("Не удалось подготовить запуск: " + error.message);
    return;
  }
  if (state.task !== task || generation !== state.launchGeneration || viewGeneration !== state.generation) return;
  state.profiles.set(task, profile);
  const selectedValidation = launchValidationConfig(action, profile);
  let author = "";
  try {
    author = localStorage.getItem("alfa-author") || "";
  } catch {}
  $("#launch-form").innerHTML = `<div class="recipe-options">${[
    ["baseline", "Baseline", "Стартовая точка · без API"],
    ["extra_trees", "ExtraTrees", "Сравнение с baseline · без API"],
    ["agent", "Qwen", "Пишет код · общий API-бюджет"],
  ]
    .map(
      ([value, label, desc]) =>
        `<label class="recipe-option"><input type="radio" name="recipe" value="${value}" ${value === action ? "checked" : ""} ${value === "agent" && !ready ? "disabled" : ""}><strong>${label}</strong><small>${desc}</small></label>`,
    )
    .join(
      "",
    )}</div>${validationFields(profile, selectedValidation)}<div class="form-grid"><label class="form-field">Ваше имя<input name="author" maxlength="80" value="${esc(author)}" placeholder="Необязательно" autocomplete="name"></label><label class="form-field" id="qwen-time-field" ${action === "agent" ? "" : "hidden"}>Время исследования Qwen, минут<input name="minutes" type="number" min="1" max="175" value="20" ${action === "agent" ? "" : "disabled"} required></label><label class="form-field full"><span id="hypothesis-label">${action === "agent" ? "Гипотеза для Qwen" : "Заметка к эксперименту"}</span><textarea name="hypothesis" maxlength="500" placeholder="Необязательно"></textarea></label></div><p class="launch-note">Каждый запуск сохранит свой результат. Текущий CSV изменится только после выбора «Использовать CSV».</p><p id="launch-error" class="form-error" role="alert"></p><div class="form-actions"><span id="launch-cost" class="form-cost">${action === "agent" ? "Qwen · расходует общий API-бюджет" : "Без API · обучение на этом компьютере"}</span><button class="button primary" type="submit" id="launch-submit">Запустить ${icon("activity")}</button></div>`;
  $("#launch-form").onchange = updateLaunchFields;
  updateLaunchFields();
  $("#launch-dialog").showModal();
}
async function startPrepare() {
  const button = $('[data-launch="prepare"]');
  if (button) button.disabled = true;
  try {
    const response = await post("/api/jobs", {
      task_id: null,
      action: "prepare",
      author: "",
      hypothesis: "",
      minutes: 20,
    });
    state.task = null;
    state.workflow = null;
    $("#task-select").innerHTML = "<option>Подготовка данных</option>";
    $("#task-select").disabled = true;
    state.selectedJob = response.job.id;
    state.runMode = "jobs";
    state.view = "runs";
    history.replaceState(null, "", "#runs");
    await renderView();
  } catch (error) {
    toast(error.message);
  } finally {
    if (button?.isConnected) button.disabled = false;
  }
}
$("#launch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget,
    button = $("#launch-submit");
  if (button.disabled) return;
  const action = $('input[name="recipe"]:checked', form)?.value,
    author = form.elements.author.value.trim(),
    hypothesis = form.elements.hypothesis.value.trim(),
    minutes = action === "agent" ? Number(form.elements.minutes.value) : 20,
    task = state.launchTask;
  if (!action || !Number.isInteger(minutes) || minutes < 1 || minutes > 175) {
    $("#launch-error").textContent = "Укажите целое число минут от 1 до 175.";
    return;
  }
  let validation;
  if (["baseline", "extra_trees"].includes(action)) {
    const strategy = $('input[name="validation_strategy"]:checked', form)?.value;
    if (!["random", "time", "grouped"].includes(strategy)) {
      $("#launch-error").textContent = "Выберите способ проверки качества.";
      return;
    }
    validation = { strategy };
    if (strategy === "time") {
      const column = form.elements.time_column.value;
      if (!column) {
        $("#launch-error").textContent = "Выберите столбец периода.";
        return;
      }
      validation.time_column = column;
    }
  }
  button.disabled = true;
  $("#launch-error").textContent = "";
  try {
    const response = await post("/api/jobs", {
      task_id: task,
      action,
      author,
      hypothesis,
      minutes,
      ...(validation ? { validation } : {}),
    });
    try {
      localStorage.setItem("alfa-author", author);
    } catch {}
    $("#launch-dialog").close();
    state.selectedJob = response.job.id;
    state.runMode = "jobs";
    state.jobDetail = null;
    if (state.task === task) {
      state.view = "runs";
      history.replaceState(null, "", "#runs");
      await renderView();
    } else toast("Эксперимент запущен.");
  } catch (error) {
    $("#launch-error").textContent = error.message;
  } finally {
    if (button.isConnected) button.disabled = false;
  }
});
async function stopJob() {
  const id = state.selectedJob,
    task = state.task,
    generation = state.generation,
    button = $('[data-action="stop-job"]');
  if (!id) return;
  if (button) button.disabled = true;
  try {
    await post(`/api/jobs/${encodeURIComponent(id)}/stop`, { task_id: task });
    if (
      state.task === task &&
      state.selectedJob === id &&
      state.generation === generation &&
      state.view === "runs"
    )
      await loadJobsPage(state.generation, true);
  } catch (error) {
    toast(error.message);
    if (button?.isConnected) button.disabled = false;
  }
}
function showPublish(id) {
  const job =
    state.jobs.find((j) => j.id === id) ||
    state.experimentReports.find((j) => j.id === id) ||
    state.jobDetail?.job;
  if (
    !job ||
    job.id !== id ||
    job.source === "imported" ||
    job.status !== "completed" ||
    job.output_valid !== true
  )
    return;
  const current =
    state.workflow?.dataset_fingerprint || state.currentDatasetFingerprint;
  if (current && job.dataset_fingerprint !== current) {
    toast("Этот CSV относится к другой версии данных.");
    return;
  }
  state.publishJob = { id, task: state.task };
  $("#publish-content").innerHTML =
    `<div class="publish-summary"><strong>${esc(job.model || ACTION_NAMES[job.action] || "Эксперимент")}</strong>${job.metric && Number.isFinite(job.metric.value) ? `<p>${esc(job.metric.name)}: ${number(job.metric.value, 6)}</p>` : ""}<p>Этот файл станет текущим submission.csv задачи. ${state.workflow?.submission?.exists ? "Существующий CSV будет заменён." : "Исходные данные останутся без изменений."}</p></div><p id="publish-error" class="form-error" role="alert"></p><div class="form-actions"><button class="button" data-close-dialog="publish-dialog">Отмена</button><button class="button primary" data-action="confirm-publish">Использовать CSV</button></div>`;
  $("#publish-dialog").showModal();
}
async function publishJob() {
  const selected = state.publishJob,
    generation = state.generation,
    button = $('[data-action="confirm-publish"]');
  if (!selected || button.disabled) return;
  button.disabled = true;
  try {
    await post(`/api/jobs/${encodeURIComponent(selected.id)}/publish`, {
      task_id: selected.task,
    });
    $("#publish-dialog").close();
    toast("CSV выбран для задачи");
    if (state.task === selected.task && generation === state.generation) {
      const workflow = await api(`${apiBase(selected.task)}/workflow`);
      if (state.task !== selected.task || generation !== state.generation)
        return;
      state.workflow = workflow;
      if (state.view === "result") await loadResult(generation);
      else if (state.view === "runs") await loadJobsPage(generation, true);
      if (generation === state.generation) mountWorkflow();
    }
  } catch (error) {
    $("#publish-error").textContent = error.message;
    if (button.isConnected) button.disabled = false;
  }
}
function comparisonWinners(reports) {
  const groups = new Map();
  for (const r of reports) {
    if (
      r.source !== "local" ||
      r.status !== "completed" ||
      r.dataset_fingerprint !== state.currentDatasetFingerprint ||
      !r.comparison_key ||
      !r.dataset_fingerprint ||
      !r.metric ||
      !Number.isFinite(r.metric.value) ||
      typeof r.metric.higher_is_better !== "boolean"
    )
      continue;
    const key = [
      r.dataset_fingerprint,
      r.comparison_key,
      r.metric.name,
      r.metric.higher_is_better,
    ].join("|");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  const winners = new Set();
  let groupCount = 0;
  for (const rows of groups.values()) {
    if (rows.length < 2) continue;
    groupCount++;
    const best = rows.reduce((a, b) =>
      (
        b.metric.higher_is_better
          ? b.metric.value > a.metric.value
          : b.metric.value < a.metric.value
      )
        ? b
        : a,
    );
    rows
      .filter((r) => r.metric.value === best.metric.value)
      .forEach((r) => winners.add(r.id));
  }
  return { winners, groupCount };
}
function experimentsPanel(reports) {
  const { winners, groupCount } = comparisonWinners(reports);
  return `<section class="panel" id="experiment-history"><div class="panel-head result-history-head"><div><h2>Эксперименты команды</h2><p>Метрики, гипотезы и авторы</p></div><div class="action-buttons"><button class="button small" data-launch="extra_trees">Сравнить модели</button></div></div>${reports.length ? `<div class="table-wrap experiment-table"><table aria-label="Результаты экспериментов"><thead><tr><th scope="col">Модель и гипотеза</th><th scope="col">Метрика</th><th scope="col">Проверка</th><th scope="col">Автор</th><th scope="col">Время / статус</th><th scope="col">Результат</th></tr></thead><tbody>${reports.map((r) => `<tr class="${r.source === "imported" ? "experiment-row-imported" : winners.has(r.id) ? "experiment-row-best" : ""}"><td><span class="model-name">${esc(r.model || ACTION_NAMES[r.action] || "Эксперимент")}</span><span class="report-hypothesis">${esc(r.hypothesis || "Гипотеза не указана")}</span><span class="report-source ${r.source === "imported" ? "source-imported" : ""}">${r.source === "imported" ? "От участника · не проверено локально" : "На этом компьютере"}</span>${r.dataset_fingerprint && state.currentDatasetFingerprint && r.dataset_fingerprint !== state.currentDatasetFingerprint ? badge("Другая версия данных", "amber") : ""}</td><td class="metric-cell">${r.metric && Number.isFinite(r.metric.value) ? number(r.metric.value, 6) : "—"}<small>${esc(r.metric?.name || "Нет метрики")}</small>${winners.has(r.id) ? badge("Лучший в группе", "green") : ""}</td><td class="validation-cell">${validationView(r, true)}</td><td>${esc(r.author || "Не указан")}</td><td>${duration(r.elapsed_seconds)}<br>${statusBadge(r.agent_status === "fallback" ? "fallback" : r.status)}</td><td>${r.source === "local" && r.status === "completed" && r.output_valid === true && r.dataset_fingerprint === state.currentDatasetFingerprint ? `<div class="result-row-actions">${r.published ? badge("CSV используется", "green") : `<button class="button small" data-action="publish-job" data-publish-id="${esc(r.id)}">Использовать CSV</button>`}${jobBundleLink(r, "CSV и отчёт ZIP", true)}</div>` : r.source === "imported" ? '<span class="subtle-text">Только отчёт</span>' : '<span class="subtle-text">—</span>'}</td></tr>`).join("")}</tbody></table></div><div class="comparison-note">${groupCount ? `${groupCount} групп сравнения: одна задача, схема, метрика и одинаковые проверочные строки.` : "Пока нет пары локальных результатов с одинаковой схемой проверки."} Метрики разных схем не ранжируются вместе.</div>` : '<div class="job-empty"><p>Запустите baseline — появится первая метрика для сравнения.</p><button class="button primary" data-launch="baseline">Обучить baseline</button></div>'}</section>`;
}
function downloadsPanel() {
  return `<section class="panel downloads-panel"><div class="panel-head"><div><h2>Выгрузить по кнопке</h2><p>Выберите, кому нужен результат</p></div>${icon("download")}</div><div class="download-options"><div class="download-option"><span class="download-kind">КОМАНДЕ</span><h3>Отчёт об экспериментах</h3><p>Метрики, авторы и заметки. Можно добавить JSON или ZIP-пакет коллеги.</p><div class="action-buttons"><a class="button" href="${apiBase()}/export" download>${icon("download")} Скачать отчёт JSON</a><button class="button" data-action="import-reports">Добавить отчёт коллеги</button></div></div><div class="download-option"><span class="download-kind">НА ХАКАТОН</span><h3>Архив с агентом</h3><p>Код решения для сдачи. Данные и ключи в архив не входят.</p><button class="button primary" data-action="build-delivery" ${state.deliveryBusy ? "disabled" : ""}>${state.deliveryBusy ? '<span class="spinner"></span> Собираем…' : `${icon("download")} Собрать solution.zip`}</button></div></div></section>`;
}
async function downloadDelivery(button) {
  if (state.deliveryBusy) return;
  state.deliveryBusy = true;
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Собираем…';
  try {
    const response = await fetch("/api/delivery", {
      method: "POST",
      cache: "no-store",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/zip",
        "X-Alfa-Token": state.session?.token || "",
      },
      body: "{}",
    });
    if (!response.ok) {
      let error;
      try {
        error = await response.json();
      } catch {}
      throw new Error(
        error?.error || `Не удалось собрать архив: ${response.status}`,
      );
    }
    if (!response.headers.get("Content-Type")?.includes("application/zip"))
      throw new Error(
        "Сервер не вернул ZIP. Перезапустите обновлённое приложение.",
      );
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "solution.zip";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("Архив решения готов. Скачивание началось.");
  } catch (error) {
    toast(error.message || "Не удалось скачать архив решения.");
  } finally {
    state.deliveryBusy = false;
    $$('[data-action="build-delivery"]').forEach((control) => {
      control.disabled = false;
      control.innerHTML = `${icon("download")} Собрать solution.zip`;
    });
  }
}
async function loadResult(generation) {
  const task = state.task;
  await loadLegacyResult(generation);
  if (generation !== state.generation) return;
  // Replace the old empty quality-history card with actual experiment reports.
  const legacy = [...main.children].find((el) => el.matches("section.panel"));
  if (legacy)
    legacy.outerHTML =
      '<div id="experiment-history">' + inlineLoading() + "</div>";
  $("#experiment-history")?.insertAdjacentHTML("beforebegin", downloadsPanel());
  try {
    const response = await api(`${apiBase(task)}/experiments`);
    if (generation !== state.generation) return;
    state.currentDatasetFingerprint = response.dataset_fingerprint || null;
    state.experimentReports = response.experiments || [];
    $("#experiment-history").outerHTML = experimentsPanel(
      state.experimentReports,
    );
  } catch (error) {
    if (generation === state.generation && $("#experiment-history"))
      $("#experiment-history").innerHTML =
        `<div class="warning-box">${esc(error.message)}</div>`;
  }
  mountWorkflow();
}
$("#report-file").addEventListener("change", async (event) => {
  const input = event.currentTarget,
    file = input.files?.[0],
    task = state.task,
    generation = state.generation;
  input.value = "";
  if (!file || !task) return;
  const zipped = /\.zip$/i.test(file.name);
  if (file.size > (zipped ? 20 : 1) * 1024 * 1024) {
    toast(
      zipped
        ? "Пакет слишком большой. Максимум 20 МБ."
        : "Отчёт слишком большой. Максимум 1 МБ.",
    );
    return;
  }
  input.disabled = true;
  toast("Добавляем отчёт…");
  try {
    const result = zipped
      ? await api(`${apiBase(task)}/import`, {
          method: "POST",
          headers: { "Content-Type": "application/zip" },
          body: file,
        })
      : await post(`${apiBase(task)}/import`, JSON.parse(await file.text()));
    toast(`Добавлено: ${result.imported}; пропущено: ${result.skipped}`);
    if (task === state.task && generation === state.generation)
      await loadResult(generation);
  } catch (error) {
    toast(
      error instanceof SyntaxError
        ? "Выберите JSON-отчёт или ZIP-пакет запуска из Alfa Lab."
        : error.message,
    );
  } finally {
    input.disabled = false;
  }
});
async function pollWorkspace() {
  if (document.hidden || state.jobPollBusy || state.pollBusy) return;
  const generation = state.generation,
    task = state.task;
  if (state.view === "runs") {
    state.jobPollBusy = true;
    try {
      if (state.runMode === "agent") {
        state.pollBusy = true;
        await loadLegacyRuns(generation, true);
        if (generation === state.generation && !$("#run-view-tabs"))
          $(".summary-strip")?.insertAdjacentHTML("afterend", runTabs());
      } else await loadJobsPage(generation, true);
      if (task && Date.now() - state.workflowUpdated > 3000) {
        const workflow = await api(`${apiBase(task)}/workflow`);
        if (generation === state.generation) {
          state.workflow = workflow;
          state.workflowUpdated = Date.now();
          mountWorkflow();
        }
      }
    } catch {
      const node = $("#job-refresh-time") || $("#last-refresh");
      if (node) node.textContent = "Нет обновления. Повторим автоматически.";
    } finally {
      state.jobPollBusy = false;
      state.pollBusy = false;
    }
  } else if (task && Date.now() - state.workflowUpdated > 3000) {
    state.jobPollBusy = true;
    try {
      const workflow = await api(`${apiBase(task)}/workflow`);
      if (generation !== state.generation) return;
      state.workflow = workflow;
      state.workflowUpdated = Date.now();
      mountWorkflow();
      if (state.view === "data" && $(".next-action"))
        $(".next-action").outerHTML = actionPanel();
      if (state.view === "result") {
        const response = await api(`${apiBase(task)}/experiments`);
        if (generation !== state.generation) return;
        const fingerprint = JSON.stringify([
          response.dataset_fingerprint,
          response.experiments,
        ]);
        if (
          fingerprint !== state.experimentFingerprint &&
          $("#experiment-history")
        ) {
          state.experimentFingerprint = fingerprint;
          state.currentDatasetFingerprint =
            response.dataset_fingerprint || null;
          state.experimentReports = response.experiments || [];
          $("#experiment-history").outerHTML = experimentsPanel(
            state.experimentReports,
          );
        }
      }
    } catch {
    } finally {
      state.jobPollBusy = false;
    }
  }
}
function legacyEndpoint(runId = null, fileName = null) {
  const base = state.inspectJob
    ? `/api/jobs/${encodeURIComponent(state.inspectJob)}/agent-runs`
    : apiBase() + "/runs";
  const params = new URLSearchParams();
  if (state.inspectJob) params.set("task_id", state.task);
  if (fileName !== null) params.set("name", fileName);
  return (
    base +
    (runId ? "/" + encodeURIComponent(runId) : "") +
    (fileName !== null ? "/file" : "") +
    (params.toString() ? "?" + params.toString() : "")
  );
}
function openJob(id, task) {
  state.task = task;
  state.selectedJob = id;
  state.jobDetail = null;
  state.workflow = null;
  state.inspectJob = null;
  state.runMode = "jobs";
  state.view = "runs";
  if (task) {
    if (![...$("#task-select").options].some((option) => option.value === task))
      $("#task-select").innerHTML = state.tasks
        .map(
          (item) =>
            `<option value="${esc(item.id)}">${esc(item.name)}</option>`,
        )
        .join("");
    $("#task-select").value = task;
    $("#task-select").disabled = false;
  } else {
    $("#task-select").innerHTML = "<option>Подготовка данных</option>";
    $("#task-select").disabled = true;
  }
  history.replaceState(null, "", "#runs");
  renderView();
}
