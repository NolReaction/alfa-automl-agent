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
  view: "overview",
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
  launchProfile: null,
  launchModelDrafts: {},
  launchModelRecipe: null,
  launchTrainingDevice: "cpu",
  launchFeatureConfigs: {},
  launchOriginalConfigs: {},
  trainingDevices: null,
  trainingDevicesBusy: false,
  resultData: null,
  workflowUpdated: 0,
  experimentFingerprint: null,
  inspectJob: null,
  currentDatasetFingerprint: null,
  nativeQueues: [],
  nativeActiveQueue: null,
  nativeSelectedQueue: null,
  nativeDetail: null,
  nativePreflight: null,
  nativeRegressionTask: null,
  nativeClassificationTask: null,
  nativeBudgetSeconds: 10500,
  nativeStartBusy: false,
  nativePreflightBusy: false,
  nativePollBusy: false,
  nativeDetailGeneration: 0,
  nativeLogGeneration: 0,
  nativeLogKind: "coordinator",
  nativeLogProfile: "regression_off",
  nativeActivityUpdated: 0,
  nativeActivityBusy: false,
  dashboard: null,
  system: null,
  overviewPollBusy: false,
  overviewStructure: null,
  campaigns: [],
  campaignActive: null,
  campaignPresets: null,
  campaignDraft: null,
  campaignRevision: 0,
  campaignPreview: null,
  campaignPreviewBody: null,
  campaignPreviewRevision: null,
  campaignPreviewBusy: false,
  campaignStartBusy: false,
  campaignSelected: null,
  campaignDetail: null,
  campaignReport: null,
  campaignDetailGeneration: 0,
  campaignPollBusy: false,
  campaignError: "",
  campaignControlBusy: null,
  campaignEditorOpen: false,
  campaignSourceJobs: [],
  campaignStudyRequest: null,
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
function taskName(task) {
  const kind = { regression: "Регрессия", classification: "Классификация" }[task?.task_type];
  return `${task?.name || "Задача"}${kind ? " · " + kind : ""}`;
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

// Workbench: current machine readings and deliberate, reviewable test series.
const CAMPAIGN_ACTION_NAMES = {automl: "Автоподбор", baseline: "Базовая модель", extra_trees: "ExtraTrees", catboost: "CatBoost", lightgbm: "LightGBM", linear: "Линейная модель"};
const CAMPAIGN_STATES = {
  queued: ["В очереди", "gray"], pending: ["Ещё не запускалось", "gray"], launching: ["Запускается", "amber"], running: ["Выполняется", "red"],
  pausing: ["Пауза после текущего задания", "amber"], paused: ["На паузе", "amber"],
  stopping: ["Останавливается", "amber"], cleanup_pending: ["Остановка требует проверки", "amber"],
  completed: ["Завершено", "green"], stopped: ["Остановлено", "gray"], cancelled: ["Не выполнено", "gray"],
  failed: ["Ошибка", "red"], interrupted: ["Прервано", "amber"], skipped: ["Пропущено", "gray"],
};
function campaignStatus(status) {
  const [label, tone] = CAMPAIGN_STATES[status] || ["Статус неизвестен", "gray"];
  return badge(label, tone);
}
function campaignRunning(campaign) {
  return Boolean(campaign && ["queued", "running", "pausing", "paused", "stopping", "cleanup_pending"].includes(campaign.status));
}
function setCampaignActive(campaign) {
  state.campaignActive = campaignRunning(campaign) ? campaign : null;
  applyNativeLaneControls();
}
function campaignPath(id) {
  if (typeof id !== "string" || !/^[0-9a-f]{32}$/.test(id)) throw new Error("Неверный идентификатор серии.");
  return `/api/campaigns/${encodeURIComponent(id)}`;
}
function readyTasks() { return state.tasks.filter((task) => task.available === true); }
function displayTask(id) { return state.tasks.find((task) => task.id === id)?.name || id || "Задача"; }
function systemPercent(value) { return Number.isFinite(value) ? percent(value) : "Нет измерения"; }
function gib(value) { return Number.isFinite(value) ? `${number(value / 1024 ** 3, 1)} ГиБ` : "Нет измерения"; }
function systemMeter(value) {
  return Number.isFinite(value) ? `<meter class="system-meter" min="0" max="100" value="${Math.max(0, Math.min(100, value))}" aria-label="Загрузка ${number(value)} процентов">${number(value)}%</meter>` : "";
}
function systemHTML(system) {
  if (!system) return '<div class="workbench-empty"><p>Измерения компьютера пока недоступны. Обновите страницу или проверьте, что приложение запущено.</p></div>';
  const cpu = system.cpu || {}, ram = system.memory || {}, disk = system.disk || {}, gpu = system.gpu || {};
  const devices = Array.isArray(gpu.devices) ? gpu.devices : [], sleep = system.sleep;
  return `<div class="system-grid"><article class="system-card"><span>CPU · весь компьютер</span><strong class="system-value">${systemPercent(cpu.percent)}</strong>${systemMeter(cpu.percent)}<p>${esc(cpu.name || "Название процессора не получено")}</p><small>${Number.isFinite(cpu.logical_cores) ? `${number(cpu.logical_cores)} логических ядер` : esc(cpu.reason || "Сбор измерения…")}</small></article><article class="system-card"><span>Оперативная память</span><strong class="system-value">${systemPercent(ram.percent)}</strong>${systemMeter(ram.percent)}<p>Свободно ${gib(ram.available_bytes)} из ${gib(ram.total_bytes)}</p>${ram.available === false ? `<small>${esc(ram.reason || "Измерение недоступно")}</small>` : ""}</article><article class="system-card"><span>${esc(disk.label || "Диск результатов")}</span><strong class="system-value">${gib(disk.free_bytes)}</strong><p>Свободное место · всего ${gib(disk.total_bytes)}</p>${disk.available === false ? `<small>${esc(disk.reason || "Измерение недоступно")}</small>` : ""}${disk.note ? `<small>${esc(disk.note)}</small>` : ""}</article><article class="system-card"><span>Видеокарта · весь компьютер</span>${devices.length ? devices.map((device) => `<div><strong class="system-value">${systemPercent(device.utilization_percent)}</strong><p>${esc(device.name || "GPU")}</p><small>VRAM ${gib(device.memory_used_bytes)} / ${gib(device.memory_total_bytes)}${Number.isFinite(device.temperature_c) ? ` · ${number(device.temperature_c)} °C` : ""}</small></div>`).join("") : `<strong class="system-value">Нет измерения</strong><p>${esc(gpu.reason || "Данные видеокарты недоступны")}</p>`}</article></div><p class="system-note">Это нагрузка всего компьютера, включая другие программы. Обучение в сериях использует CPU в 8 потоков; GPU для обучения не включён.${system.sampled_at ? ` Измерено ${esc(date(system.sampled_at))}.` : ""}</p>${sleep ? `<p class="system-note">${esc(sleep.active === true ? "Windows: запрос против простоя и сна активен на время работы." : sleep.requested === true ? "Windows: запрос против сна не подтверждён." : "Запрос против простоя и сна сейчас не активен.")}${sleep.reason ? ` ${esc(sleep.reason)}` : ""} Закрытие крышки и ручное выключение компьютера этим не блокируются.</p>` : ""}`;
}
function phaseName(job) {
  if (job?.status === "cleanup_pending") return "Остановка требует проверки. Повторите остановку в эксперименте.";
  return ({queued: "Ожидание запуска", preparing: "Подготовка данных", training: "Обучение и подбор моделей", validating: "Проверка качества", refitting: "Обучение выбранной модели на всех данных", saving_model: "Сохранение обученной модели", finished: "Сохранение результата"})[job?.stage] || (job?.status === "queued" ? "Ожидание запуска" : "Работа продолжается");
}
function campaignCompleted(campaign) {
  return (campaign?.steps || []).filter((step) => step.status === "completed").length;
}
function overviewActivity(data) {
  const campaign = data.active_campaign, native = data.active_queue, job = data.active_job;
  if (campaignRunning(campaign)) {
    const current = (campaign.steps || []).find((step) => ["launching", "running"].includes(step.status));
    return `<article class="activity-card"><div class="workbench-section-head"><h3>${esc(campaign.name || "Серия тестов")}</h3>${campaignStatus(campaign.status)}</div><p class="activity-phase">${campaign.status === "paused" ? "Серия ждёт явного продолжения. Старые задания повторно не запускаются." : current ? `${esc(displayTask(current.task_id))} · ${esc(phaseName(current.job))}` : "Подготовка следующего задания"}</p><p class="activity-count">Выполнено ${campaignCompleted(campaign)} из ${(campaign.steps || []).length} заданий · прошло ${duration(campaign.elapsed_seconds)}</p><button class="button primary" data-campaign-open="${esc(campaign.id)}">Открыть серию ${icon("activity")}</button></article>`;
  }
  if (nativeRunning(native)) {
    const current = (native.profiles || []).find((item) => item.status === "running");
    return `<article class="activity-card"><div class="workbench-section-head"><h3>Проверка агента</h3>${nativeStatus(native.status)}</div><p>${esc(current ? NATIVE_NAMES[current.profile] || current.profile : "Очередь проверки агента занята")}</p><button class="button primary" data-workflow-view="native">Открыть проверку</button></article>`;
  }
  if (runningJob(job)) return `<article class="activity-card"><div class="workbench-section-head"><h3>${esc(CAMPAIGN_ACTION_NAMES[job.action] || "Эксперимент")}</h3>${statusBadge(job.status)}</div><p class="activity-phase">${esc(phaseName(job))}</p><p class="activity-count">${esc(displayTask(job.task_id))} · прошло ${duration(job.elapsed_seconds)}</p><button class="button primary" data-open-job="${esc(job.id)}" data-task-id="${esc(job.task_id || "")}">Открыть эксперимент</button></article>`;
  return '<div class="workbench-empty"><h3>Компьютер свободен для следующего теста</h3><p>Выберите готовый план серии. Приложение выполнит задания последовательно и сохранит отчёты.</p><button class="button primary" data-workflow-view="campaigns">Собрать серию тестов</button></div>';
}
function overviewHTML(data, system) {
  const tasks = Array.isArray(data.tasks) ? data.tasks : state.tasks;
  const ready = tasks.filter((task) => task.available === true), busy = Boolean(data.active_job || data.active_queue || data.active_campaign);
  const recent = (data.recent_jobs || []).slice(0, 4);
  return heading("ЛОКАЛЬНАЯ ЛАБОРАТОРИЯ", "Обзор", "Подготовьте данные, выберите план и оставьте последовательные тесты работать на компьютере.", false) +
    `<section class="overview-hero"><div class="overview-hero-copy"><span class="eyebrow">СЛЕДУЮЩИЙ ШАГ</span><h2>${ready.length ? busy ? "Следите за текущей работой" : "Запустите осмысленную серию тестов" : "Подготовьте данные для первых тестов"}</h2><p>${ready.length ? "Короткая проверка, сравнение пяти моделей или дневной автоподбор с разными лимитами." : "Загрузите публичные задачи. После загрузки они появятся в выборе серии."}</p><div class="overview-actions">${ready.length ? '<button class="button primary" data-workflow-view="campaigns">Выбрать план тестов ' + icon("clock") + '</button><button class="button" data-workflow-view="runs">Один эксперимент</button>' : '<button class="button primary" data-launch="prepare">Загрузить публичные данные</button>'}</div></div><div class="overview-hero-fact"><strong>${ready.length} / ${tasks.length}</strong><span>задач готовы</span><small>Без API · CPU, 8 потоков</small></div></section>` +
    `<div class="overview-grid"><section class="panel overview-card"><div class="panel-head"><div><h2>Сейчас выполняется</h2><p>Текущий этап и сохранённый прогресс</p></div><span id="overview-refresh" class="subtle-text">Обновляется автоматически</span></div><div class="panel-body" id="overview-activity">${overviewActivity(data)}</div></section><section class="panel overview-card"><div class="panel-head"><div><h2>Данные для тестов</h2><p>Готовность файлов перед обучением</p></div></div><div class="overview-task-list">${tasks.length ? tasks.map((task) => `<div class="overview-task-item"><div><strong>${esc(taskName(task))}</strong><small>${task.available ? "Файлы готовы к проверке и обучению" : esc(task.error || "Файлы ещё не загружены")}</small></div>${badge(task.available ? "Готово" : "Нужны данные", task.available ? "green" : "amber")}<button class="button small" data-overview-task="${esc(task.id)}">Открыть</button></div>`).join("") : '<div class="workbench-empty"><p>Задач пока нет.</p><button class="button" data-launch="prepare">Загрузить данные</button></div>'}</div></section></div>` +
    `<section class="panel overview-card"><div class="panel-head"><div><h2>Ваш компьютер</h2><p>Фактические измерения системы</p></div>${badge("CPU-обучение", "gray")}</div><div class="panel-body" id="overview-system">${systemHTML(system)}</div></section>` +
    `<section class="panel overview-card"><div class="panel-head"><div><h2>Последние эксперименты</h2><p>Результаты сохраняются после завершения приложения</p></div><button class="text-button" data-workflow-view="result">Все результаты →</button></div><div class="activity-list">${recent.length ? recent.map((job) => `<div class="overview-task-item"><div><strong>${esc(CAMPAIGN_ACTION_NAMES[job.action] || "Эксперимент")} · ${esc(displayTask(job.task_id))}</strong><small>${esc(date(job.started_at))} · ${duration(job.elapsed_seconds)}</small></div>${statusBadge(job.status)}<span>${esc(nativeMetric(job.metric))}</span><button class="button small" data-open-job="${esc(job.id)}" data-task-id="${esc(job.task_id || "")}">Открыть</button></div>`).join("") : '<div class="workbench-empty"><p>После первого запуска здесь появятся настоящие измерения качества.</p></div>'}</div></section>`;
}
async function loadOverview(generation, quiet = false) {
  const [data, system] = await Promise.all([api("/api/dashboard"), api("/api/system").catch(() => null)]);
  if (generation !== state.generation || state.view !== "overview") return;
  state.dashboard = data; state.system = system;
  if (Array.isArray(data.tasks)) state.tasks = data.tasks;
  state.session = {...state.session, active_job: data.active_job, active_campaign: data.active_campaign};
  setNativeActive(data.active_queue); setCampaignActive(data.active_campaign);
  const structure = JSON.stringify({tasks: data.tasks, recent: data.recent_jobs, busy: Boolean(data.active_job || data.active_queue || data.active_campaign)});
  if (!quiet || structure !== state.overviewStructure || !$("#overview-activity")) main.innerHTML = overviewHTML(data, system);
  else {
    $("#overview-activity").innerHTML = overviewActivity(data);
    $("#overview-system").innerHTML = systemHTML(system);
    $("#overview-refresh").textContent = `Обновлено ${new Date().toLocaleTimeString("ru-RU")}`;
  }
  state.overviewStructure = structure;
  applyNativeLaneControls();
}
function campaignPreset(id) { return (state.campaignPresets?.presets || []).find((preset) => preset.id === id); }
function lightgbmStudySource(job, taskId) {
  return job?.source === "local" && job.task_id === taskId && job.status === "completed" &&
    job.output_valid === true && ["lightgbm", "automl"].includes(job.action) &&
    /^LGBM(Classifier|Regressor)$/.test(job.model_config?.model || "") &&
    Number.isInteger(job.model_config?.parameters?.n_estimators) && job.model_config.parameters.n_estimators > 0;
}
function lightgbmStudySources(taskId) {
  const jobs = [...(state.campaignSourceJobs || []), ...state.jobs, ...state.experimentReports];
  const seen = new Set();
  return jobs.filter((job) => lightgbmStudySource(job, taskId) && !seen.has(job.id) && seen.add(job.id));
}
function lightgbmStudySteps(taskIds, sourceId = null, minutes = 30) {
  const task_id = taskIds[0];
  if (!task_id) return [];
  const source = lightgbmStudySources(task_id).find((job) => job.id === sourceId);
  const base = source?.model_config.parameters.n_estimators || 500;
  const counts = [...new Set([base, Math.min(2000, Math.ceil(base * 1.5)), Math.min(2000, base * 2)])];
  return counts.map((count, index) => ({task_id, action: "lightgbm", minutes,
    validation: structuredClone(source?.validation || {strategy: "auto"}),
    model_params: {...source?.model_config.parameters, n_estimators: count},
    ...(source?.feature_config ? {feature_config: structuredClone(source.feature_config)} : {}),
    is_control: index === 0,
    label: `${index === 0 ? "Контроль" : "Вариант " + index} · LightGBM · ${count} деревьев`,
  }));
}
function openLightgbmStudy(sourceId = null, taskId = state.task) {
  if (!readyTasks().some((task) => task.id === taskId)) { toast("Выберите готовую задачу."); return; }
  if (sourceId && !lightgbmStudySources(taskId).some((job) => job.id === sourceId)) {
    toast("Выберите завершённый локальный LightGBM с сохранёнными настройками."); return;
  }
  state.campaignStudyRequest = {task_id: taskId, source_job_id: sourceId};
  invalidateCampaignPreview();
  navigate("campaigns");
}
function updateLightgbmStudy(sourceId, minutes) {
  const draft = state.campaignDraft;
  if (draft?.preset !== "lightgbm_trees") return;
  draft.source_job_id = sourceId || null;
  draft.minutes_per_step = Number(minutes);
  draft.steps = lightgbmStudySteps(draft.task_ids, draft.source_job_id, draft.minutes_per_step);
  invalidateCampaignPreview(); renderCampaignEditor();
}
function lightgbmStudySetupHTML() {
  const draft = state.campaignDraft;
  if (draft?.preset !== "lightgbm_trees") return "";
  const sources = lightgbmStudySources(draft.task_ids[0]), source = sources.find((job) => job.id === draft.source_job_id);
  return `<section class="training-study-setup" aria-label="Настройки сравнения обучения"><div><h3>Проверяем одну гипотезу</h3><p>Поможет ли большее число деревьев? Приложение последовательно обучит контроль и более крупные варианты на одинаковых проверочных строках.</p></div><div class="form-grid"><label class="form-field">Отправная точка<select id="campaign-source-job"><option value="">Базовый рецепт · 500 / 750 / 1000 деревьев</option>${sources.map((job) => `<option value="${esc(job.id)}" ${source?.id === job.id ? "selected" : ""}>${esc(job.hypothesis || jobTitle(job))} · ${job.model_config.parameters.n_estimators} деревьев · ${esc(date(job.started_at))}</option>`).join("")}</select></label><label class="form-field">Лимит каждого варианта, минут<input id="campaign-minutes-per-step" type="number" min="1" max="175" step="1" value="${esc(draft.minutes_per_step || 30)}"></label></div><p class="campaign-note">${source ? "Сохраняем параметры, обработку признаков и схему проверки выбранного запуска; меняется только число деревьев. Контроль обучается заново." : "Используется базовый рецепт приложения. Он не считается точной копией победителя вашего ночного прогона. После первого LightGBM можно выбрать его здесь."}</p><p class="campaign-note">Это сравнение рецептов на локальных данных. Оно помогает выбрать следующую проверку; поведение полного конкурсного агента проверяется отдельно.</p></section>`;
}
function campaignStudyPreviewHTML(preview) {
  if (preview?.experiment?.kind !== "lightgbm_tree_count") return "";
  return `<section class="training-study-plan"><h3>Что именно будет обучаться</h3><p>${esc(preview.experiment.control_description || "Контроль и варианты используют одинаковые данные, обработку признаков и проверочные строки.")}</p><div class="table-wrap" tabindex="0" role="region" aria-label="Проверенные варианты обучения"><table><thead><tr><th>Вариант</th><th>Деревьев</th><th>Лимит исполнения</th><th>Проверка и настройки</th></tr></thead><tbody>${(preview.steps || []).map((step) => `<tr><th scope="row">${esc(step.label || "LightGBM")}${step.is_control ? badge("Контроль", "gray") : ""}</th><td>${number(step.model_config?.parameters?.n_estimators ?? step.model_params?.n_estimators)}</td><td>${duration(step.worker_timeout_seconds)}</td><td>${esc(validationLabel({validation: step.validation}))}${modelParameterView({action: step.action, model_config: step.model_config || {parameters: step.model_params}} , true)}${searchFeatureView(step.feature_config)}${step.feature_config ? `<details class="model-parameters compact"><summary>Обработка признаков</summary><dl>${Object.entries(step.feature_config).map(([name, value]) => `<dt>${esc(name)}</dt><dd>${esc(JSON.stringify(value))}</dd>`).join("")}</dl></details>` : ""}</td></tr>`).join("")}</tbody></table></div><p class="campaign-note">Каждый вариант проходит проверку качества и отдельное финальное обучение на полном train. Сохранённая модель относится к финальному обучению; метрика получена до него.</p></section>`;
}
function campaignTaskOptions(selected) {
  return readyTasks().map((task) => `<option value="${esc(task.id)}" ${selected === task.id ? "selected" : ""}>${esc(taskName(task))}</option>`).join("");
}
function campaignDefaultSteps(presetId, taskIds) {
  const preset = campaignPreset(presetId);
  if (presetId === "lightgbm_trees") return lightgbmStudySteps(taskIds, state.campaignDraft?.source_job_id, state.campaignDraft?.minutes_per_step || 30);
  if (!preset || !Array.isArray(preset.actions) || !Array.isArray(preset.budgets_minutes)) return [];
  return preset.budgets_minutes.flatMap((minutes) => preset.actions.flatMap((action) => taskIds.map((task_id) => ({task_id, action, minutes, validation: {strategy: action === "automl" ? "auto" : "random"}}))));
}
function newCampaignDraft() {
  const task_ids = readyTasks().slice(0, 2).map((task) => task.id), preset = campaignPreset("day") ? "day" : state.campaignPresets?.presets?.[0]?.id || "custom";
  return {preset, task_ids, name: campaignPreset(preset)?.name || "Моя серия тестов", max_hours: 12, steps: campaignDefaultSteps(preset, task_ids)};
}
function invalidateCampaignPreview() {
  state.campaignRevision++; state.campaignPreview = null; state.campaignPreviewBody = null; state.campaignPreviewRevision = null;
  state.campaignError = "";
}
function campaignValidationOptions(step) {
  const task = state.tasks.find((task) => task.id === step.task_id), profile = state.profiles.get(step.task_id);
  const options = [["random", "Случайные 80/20"]];
  if (step.action === "automl") options.unshift(["auto", "Автоматически"]);
  if (task?.task_type === "classification") options.push(["grouped", "Разделить дубли"]);
  if (task?.task_type === "regression" && (profile?.columns || []).some((column) => column.role === "feature" && column.dtype === "numeric")) options.push(["time", "Последний период"]);
  return options;
}
function campaignStepEditor(step, index) {
  const numeric = (state.profiles.get(step.task_id)?.columns || []).filter((column) => column.role === "feature" && column.dtype === "numeric");
  return `<li class="campaign-step"><span class="campaign-step-number">${index + 1}</span><div class="campaign-step-fields"><label class="form-field">Задача<select data-campaign-index="${index}" data-campaign-field="task_id">${campaignTaskOptions(step.task_id)}</select></label><label class="form-field">Метод<select data-campaign-index="${index}" data-campaign-field="action">${Object.entries(CAMPAIGN_ACTION_NAMES).map(([action, name]) => `<option value="${action}" ${step.action === action ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></label>${step.action === "automl" ? `<label class="form-field">Поиск<select data-campaign-index="${index}" data-campaign-field="search_strategy"><option value="adaptive" ${(step.search_strategy || "adaptive") === "adaptive" ? "selected" : ""}>Адаптивный</option><option value="legacy" ${step.search_strategy === "legacy" ? "selected" : ""}>Прежний фиксированный</option></select></label>` : ""}<label class="form-field">Лимит, минут<input type="number" min="1" max="175" step="1" data-campaign-index="${index}" data-campaign-field="minutes" value="${esc(step.minutes)}"></label><label class="form-field">Проверка качества<select data-campaign-index="${index}" data-campaign-field="strategy">${campaignValidationOptions(step).map(([value, label]) => `<option value="${value}" ${step.validation?.strategy === value ? "selected" : ""}>${esc(label)}</option>`).join("")}</select></label>${step.validation?.strategy === "time" ? `<label class="form-field">Столбец периода<select data-campaign-index="${index}" data-campaign-field="time_column"><option value="">Выберите столбец</option>${numeric.map((column) => `<option value="${esc(column.name)}" ${step.validation.time_column === column.name ? "selected" : ""}>${esc(column.name)}</option>`).join("")}</select></label>` : ""}</div><div class="campaign-step-actions"><button class="icon-button" type="button" data-campaign-move="${index}" data-direction="-1" ${index === 0 ? "disabled" : ""} aria-label="Поднять задание ${index + 1}">${icon("arrowup")}</button><button class="icon-button" type="button" data-campaign-move="${index}" data-direction="1" ${index === state.campaignDraft.steps.length - 1 ? "disabled" : ""} aria-label="Опустить задание ${index + 1}">${icon("arrowdown")}</button><button class="icon-button" type="button" data-campaign-remove="${index}" aria-label="Удалить задание ${index + 1}">×</button></div></li>`;
}
function campaignBody() {
  const draft = state.campaignDraft;
  if (!draft) throw new Error("Выберите план серии.");
  if (draft.preset === "lightgbm_trees") {
    if (draft.task_ids.length !== 1 || !readyTasks().some((task) => task.id === draft.task_ids[0])) throw new Error("Выберите одну готовую задачу для сравнения обучения.");
    if (!Number.isInteger(draft.minutes_per_step) || draft.minutes_per_step < 1 || draft.minutes_per_step > 175) throw new Error("Лимит каждого варианта — целое число минут от 1 до 175.");
    if (!Number.isFinite(draft.max_hours) || draft.max_hours <= 0 || draft.max_hours > 12) throw new Error("Общий лимит — больше 0 и не больше 12 часов.");
    if (draft.source_job_id && !lightgbmStudySources(draft.task_ids[0]).some((job) => job.id === draft.source_job_id)) throw new Error("Исходный LightGBM недоступен. Выберите другой запуск или базовый рецепт.");
    if (draft.source_job_id && lightgbmStudySources(draft.task_ids[0]).find((job) => job.id === draft.source_job_id)?.model_config.parameters.n_estimators >= 2000) throw new Error("У исходной модели уже 2000 деревьев или больше. Выберите другой контроль для увеличения числа деревьев.");
    return {preset: draft.preset, task_ids: [...draft.task_ids], name: draft.name.trim().slice(0, 100), max_hours: draft.max_hours,
      minutes_per_step: draft.minutes_per_step, ...(draft.source_job_id ? {source_job_id: draft.source_job_id} : {})};
  }
  const allowed = new Set(readyTasks().map((task) => task.id)), task_ids = [...new Set(draft.steps.map((step) => step.task_id))];
  if (task_ids.length < 1 || task_ids.length > 2 || task_ids.some((id) => !allowed.has(id))) throw new Error("Выберите одну или две готовые задачи.");
  if (draft.steps.length < 1 || draft.steps.length > 12) throw new Error("В серии должно быть от 1 до 12 заданий.");
  if (!Number.isFinite(draft.max_hours) || draft.max_hours <= 0 || draft.max_hours > 12) throw new Error("Общий лимит — больше 0 и не больше 12 часов.");
  const steps = draft.steps.map((step) => {
    if (!Object.hasOwn(CAMPAIGN_ACTION_NAMES, step.action)) throw new Error("Выберите известный метод обучения.");
    if (!Number.isInteger(step.minutes) || step.minutes < 1 || step.minutes > 175) throw new Error("Лимит задания — целое число минут от 1 до 175.");
    const strategy = step.validation?.strategy || (step.action === "automl" ? "auto" : "random");
    if (!campaignValidationOptions(step).some(([value]) => value === strategy)) throw new Error("Выберите доступный способ проверки качества.");
    const validation = {strategy};
    if (strategy === "time") {
      const column = step.validation.time_column;
      if (!(state.profiles.get(step.task_id)?.columns || []).some((item) => item.name === column && item.role === "feature" && item.dtype === "numeric")) throw new Error("Укажите числовой столбец периода.");
      validation.time_column = column;
    }
    return {task_id: step.task_id, action: step.action, minutes: step.minutes, validation, ...(step.action === "automl" ? {search_strategy: step.search_strategy || "adaptive"} : {}), ...(step.model_config ? {model_config: structuredClone(step.model_config)} : step.model_params ? {model_params: structuredClone(step.model_params)} : {}), ...(step.feature_config ? {feature_config: structuredClone(step.feature_config)} : {}), ...(typeof step.is_control === "boolean" ? {is_control: step.is_control} : {}), ...(step.label ? {label: step.label} : {})};
  });
  const body = {preset: draft.preset, task_ids, name: draft.name.trim().slice(0, 100), max_hours: draft.max_hours};
  if (draft.preset === "custom") body.steps = steps;
  return body;
}
function campaignLaunchBlock() {
  if (state.campaignStartBusy) return "Запускаем серию…";
  if (campaignRunning(state.campaignActive)) return "Текущая серия занимает очередь. Продолжите или остановите её.";
  if (nativeRunning(state.nativeActiveQueue)) return "Сейчас идёт проверка агента. Дождитесь её завершения или остановите её.";
  if (runningJob(state.session?.active_job)) return "Сейчас выполняется отдельный эксперимент.";
  if (!state.campaignPreview || state.campaignPreviewRevision !== state.campaignRevision) return "Сначала проверьте план и его лимиты.";
  if (!state.campaignPreview.ready) return "Исправьте причины, указанные в проверке плана.";
  if (!state.campaignPreview.source_seal?.valid) return "Версия исходников не подтверждена. Обновите проверку плана.";
  if (!state.campaignPreview.runtime_seal?.valid) return "Среда Python не подтверждена. Обновите проверку плана.";
  return null;
}
function campaignSealsHTML(value) {
  const source = value.source_seal || {}, runtime = value.runtime_seal || {};
  return `<div class="campaign-seals">${badge(source.valid === true ? "Код подтверждён" : source.valid === false ? "Код изменился" : "Код не проверен", source.valid === true ? "green" : "amber")}${badge(runtime.valid === true ? "Среда Python подтверждена" : runtime.valid === false ? "Среда Python изменилась" : "Среда Python не проверена", runtime.valid === true ? "green" : "amber")}${runtime.python_version ? `<span>Python ${esc(runtime.python_version)}${Number.isFinite(runtime.pointer_bits) ? ` · ${number(runtime.pointer_bits)} бит` : ""}</span>` : ""}</div>${source.valid === false || runtime.valid === false ? '<p class="campaign-error">Версия кода или среда Python изменилась. Новые результаты этой серии нельзя считать сопоставимыми; изучите отчёты и создайте новый план.</p>' : ""}`;
}
function campaignPreviewHTML() {
  const preview = state.campaignPreview;
  if (!preview) return '<div class="campaign-note"><strong>План ещё не проверен</strong><p>Проверка не обучает модели. Сервер подтвердит данные, версию кода и реальные лимиты.</p></div>';
  return `<div class="campaign-preview">${badge(preview.ready ? "План готов к запуску" : "План требует исправления", preview.ready ? "green" : "amber")}${campaignSealsHTML(preview)}<div class="campaign-preview-metrics"><div><span>Заданий</span><strong>${(preview.steps || []).length}</strong></div><div><span>Потолок всех исполнений</span><strong>${duration(preview.worker_upper_bound_seconds)}</strong></div><div><span>Общий лимит серии</span><strong>${duration(preview.wall_budget_seconds)}</strong></div></div><p>Подготовка данных и сохранение отчётов могут занять дополнительное время. Задания могут закончиться раньше лимита. Лимит каждого задания включает его обучение и формирование результата; точное ограничение показано в проверенном порядке.</p>${warningList(preview.errors)}${warningList(preview.warnings)}${campaignStudyPreviewHTML(preview)}<p class="campaign-note">Сравнимы только результаты на тех же данных и проверочных строках. Выбор лучшего по этой проверке не заменяет независимый тест и не обещает баллы платформы.</p><details><summary>Проверенный порядок и версия</summary><ol>${(preview.steps || []).map((step) => `<li>${esc(displayTask(step.task_id))} · ${esc(CAMPAIGN_ACTION_NAMES[step.action] || step.action)} · до ${duration(step.worker_timeout_seconds)} исполнения</li>`).join("")}</ol><p>Код <code>${esc(preview.source_seal?.sha || "Не подтверждён")}</code></p></details></div>`;
}
function campaignDraftPlanHTML() {
  const steps = state.campaignDraft.steps;
  const budgets = steps.map((step) => step.minutes);
  const valid = budgets.every((value) => Number.isInteger(value) && value >= 1 && value <= 175);
  return `<div class="campaign-draft-summary"><div><strong>${steps.length} последовательных заданий</strong><p>${valid && steps.length ? `До ${duration(budgets.reduce((sum, value) => sum + value, 0) * 60)} обучения по текущему плану` : "Добавьте задания и укажите лимиты"}</p></div><span>Точные лимиты подтвердит проверка плана</span></div><details class="campaign-step-editor" id="campaign-step-editor" ${state.campaignEditorOpen ? "open" : ""}><summary>Изменить задания, порядок и проверку качества</summary><ol class="campaign-step-list">${steps.map(campaignStepEditor).join("")}</ol><div class="campaign-toolbar"><button class="button small" data-action="campaign-add" ${steps.length >= 12 || !readyTasks().length ? "disabled" : ""}>+ Добавить задание</button><span class="subtle-text">${steps.length} / 12 заданий · любое изменение требует новой проверки</span></div></details>`;
}
function campaignEditorHTML() {
  const draft = state.campaignDraft, presets = state.campaignPresets?.presets || [], block = campaignLaunchBlock();
  return `<section class="panel campaign-builder"><div class="panel-head"><div><h2>План новой серии</h2><p>Выберите готовый план и при необходимости измените задания.</p></div>${badge("CPU · 8 потоков · без API", "green")}</div><div class="panel-body"><div class="campaign-presets">${presets.map((preset) => `<button class="preset-card ${draft.preset === preset.id ? "selected" : ""}" type="button" data-campaign-preset="${esc(preset.id)}" aria-pressed="${draft.preset === preset.id}"><strong>${esc(preset.name)}</strong><span>${esc(preset.description)}</span><small>${preset.id === "day" ? "30 / 90 / 175 минут на задачу · до 9 ч 50 мин для двух" : preset.id === "models" ? "Пять методов · до 30 минут каждый" : preset.id === "custom" ? "Редактируемый порядок · до 12 заданий" : preset.id === "lightgbm_trees" ? "Контроль и до двух более крупных вариантов · одна задача" : "До 5 минут на задачу"}</small></button>`).join("")}</div><fieldset class="campaign-task-picker"><legend>${draft.preset === "lightgbm_trees" ? "Задача для сравнения · выберите одну" : "Задачи серии · выберите одну или две"}</legend>${readyTasks().map((task) => `<label><input type="${draft.preset === "lightgbm_trees" ? "radio" : "checkbox"}" name="campaign-task" data-campaign-task="${esc(task.id)}" ${draft.task_ids.includes(task.id) ? "checked" : ""}> ${esc(taskName(task))}</label>`).join("") || '<p>Нет готовых задач. <button class="text-button" data-launch="prepare">Загрузить публичные данные</button></p>'}</fieldset><div class="form-grid"><label class="form-field">Название серии<input id="campaign-name" maxlength="100" value="${esc(draft.name)}"></label><label class="form-field">Общий лимит, часов<input id="campaign-max-hours" type="number" min="0.1" max="12" step="0.1" value="${esc(draft.max_hours)}"></label></div>${lightgbmStudySetupHTML()}${campaignDraftPlanHTML()}<p class="campaign-note">Последовательное обучение на CPU. Автоподбор по умолчанию адаптивный: до 256 новых вариантов в пределах времени. Отдельная модель может закончить за минуты; лимит не обещает непрерывную нагрузку. Показания GPU на обзоре относятся ко всему компьютеру.</p><div id="campaign-preview">${campaignPreviewHTML()}</div><p id="campaign-editor-error" class="form-error" role="alert">${esc(state.campaignError)}</p><div class="campaign-toolbar"><button class="button" data-action="campaign-preview" ${state.campaignPreviewBusy ? "disabled" : ""}>${state.campaignPreviewBusy ? '<span class="spinner"></span> Проверяем…' : `${icon("shield")} Проверить план`}</button><button class="button primary" data-action="campaign-start" ${block ? "disabled" : ""}>${icon("activity")} Запустить серию</button><span id="campaign-launch-block" class="subtle-text">${esc(block || "План проверен. Можно запускать.")}</span></div></div></section>`;
}
function renderCampaignEditor() {
  const node = $("#campaign-editor");
  if (node) node.innerHTML = campaignEditorHTML();
  applyNativeLaneControls();
}
function renderCampaignHistory() {
  const node = $("#campaign-history");
  if (!node) return;
  node.innerHTML = state.campaigns.length ? state.campaigns.map((campaign) => `<button class="native-history-item ${campaign.id === state.campaignSelected ? "selected" : ""}" data-campaign-open="${esc(campaign.id)}" aria-pressed="${campaign.id === state.campaignSelected}"><span><strong>${esc(campaign.name || "Серия тестов")}</strong><small>${esc(date(campaign.created_at))}</small></span>${campaignStatus(campaign.status)}</button>`).join("") : '<div class="workbench-empty"><p>Серий пока нет. Первый проверенный план появится здесь после запуска.</p></div>';
}
function comparisonRows(group) {
  if (!group?.comparison_key || !group.dataset_fingerprint || !group.metric_name || typeof group.higher_is_better !== "boolean") return [];
  return (Array.isArray(group.jobs) ? group.jobs : []).filter((job) => typeof job.job_id === "string" && Number.isFinite(job.value) && Number.isFinite(job.minutes));
}
function campaignCurve(group, rows) {
  const points = rows.filter((row) => row.action === "automl").sort((a, b) => a.minutes - b.minutes);
  if (points.length < 2) return '<p class="campaign-note">Кривая бюджета появится после двух сравнимых запусков автоподбора с разными лимитами.</p>';
  const times = points.map((row) => row.minutes), values = points.map((row) => row.value);
  const minX = Math.min(...times), rangeX = Math.max(...times) - minX;
  if (rangeX === 0) return '<p class="campaign-note">У сравнимых запусков одинаковый лимит времени.</p>';
  const minY = Math.min(...values), maxY = Math.max(...values), spanY = maxY - minY || Math.max(Math.abs(maxY) * .05, .01);
  const xy = points.map((row) => ({...row, x: 46 + (row.minutes - minX) / rangeX * 500, y: 140 - (row.value - minY) / spanY * 110}));
  return `<figure class="campaign-curve"><svg viewBox="0 0 600 190" role="img" aria-label="Локальное качество автоподбора по лимиту времени. Точные значения в таблице ниже."><path d="M46 15V140H558" fill="none" stroke="currentColor" opacity=".25"/><polyline points="${xy.map((row) => `${row.x},${row.y}`).join(" ")}" fill="none" stroke="#e5232e" stroke-width="2"/>${xy.map((row) => `<circle cx="${row.x}" cy="${row.y}" r="4" fill="#e5232e"/><text x="${row.x}" y="163" text-anchor="middle">${number(row.minutes)} мин</text>`).join("")}<text x="48" y="12">${esc(group.metric_name)} · ${group.higher_is_better ? "выше лучше" : "ниже лучше"}</text></svg><figcaption>Лимит времени, а не фактически потраченное время. Точки относятся к одной локальной проверке.</figcaption></figure>`;
}
function campaignComparisonsHTML(groups) {
  const valid = (Array.isArray(groups) ? groups : []).map((group) => ({group, rows: comparisonRows(group)})).filter(({rows}) => rows.length);
  if (!valid.length) return '<div class="workbench-empty"><h3>Сравнение появится после проверенных результатов</h3><p>Несовместимые данные и разбиения не объединяются. Ошибка процесса или непроверенный CSV не дают победителя.</p></div>';
  return valid.map(({group, rows}) => {
    const available = group.comparison_available !== false && rows.length > 1;
    const best = available ? rows.find((row) => row.job_id === group.best_job_id) : null;
    const control = rows.find((row) => row.job_id === group.control_job_id || row.is_control === true);
    const missingControl = group.comparison_reason === "control_not_admitted" || group.comparison_reason === "multiple_controls";
    const difference = (row) => available && Number.isFinite(row.improvement) ? `${row.improvement > 0 ? "+" : ""}${number(row.improvement, 6)}` : "—";
    return `<section class="campaign-comparison"><div class="workbench-section-head"><h3>${esc(displayTask(group.task_id))} · ${esc(group.metric_name)}</h3>${best ? badge((group.tied_best_job_ids || []).length > 1 ? "Одинаковый лучший результат" : "Лучший в этой локальной группе", "green") : badge(available ? "Победитель не определён" : rows.length < 2 ? "Пока один результат" : "Сравнение ещё недоступно", "gray")}</div><p class="campaign-note">${esc(validationLabel({split: group.split}))} · одна версия данных и одни проверочные строки</p>${missingControl ? '<p class="campaign-error">Контроль не прошёл обязательные проверки или определён неоднозначно. Прирост к нему не рассчитывается; другой вариант не назначается контролем автоматически.</p>' : control ? `<p class="campaign-note">Контроль: ${esc(control.label || CAMPAIGN_ACTION_NAMES[control.action] || "Первый сравнимый запуск")} · ${number(control.value, 6)}. Положительное улучшение означает более качественный результат.</p>` : ""}${available && Number.isFinite(group.gain) ? `<div class="training-comparison-summary"><strong>${group.gain > 0 ? "+" : ""}${number(group.gain, 6)}</strong><span>${group.gain > 0 ? "лучший локальный прирост к контролю" : "прироста к контролю пока нет"}</span></div>` : ""}${available && rows.some((row) => row.action === "automl") ? campaignCurve(group, rows) : !available ? '<p class="campaign-note">Для сравнения разных моделей или лимитов нужен ещё один совместимый результат.</p>' : ""}<div class="table-wrap" tabindex="0" role="region" aria-label="Сравнимые результаты серии"><table><thead><tr><th>Вариант</th><th>Лимит</th><th>Фактически</th><th>${esc(group.metric_name)}</th><th>Улучшение к контролю</th><th>Эксперимент</th></tr></thead><tbody>${rows.map((row) => `<tr class="${row.job_id === control?.job_id ? "training-control-row" : ""}"><td>${esc(row.label || CAMPAIGN_ACTION_NAMES[row.action] || row.action)} ${row.job_id === best?.job_id ? '<span aria-label="Лучший локально">★</span>' : ""}${row.job_id === control?.job_id ? badge("Контроль", "gray") : ""}</td><td>${row.minutes} мин</td><td>${Number.isFinite(row.elapsed_seconds) ? duration(row.elapsed_seconds) : "—"}</td><td>${number(row.value, 6)}</td><td class="${available && row.improvement > 0 ? "training-gain" : available && row.improvement < 0 ? "training-loss" : ""}">${difference(row)}</td><td><button class="text-button" data-open-job="${esc(row.job_id)}" data-task-id="${esc(group.task_id)}">Открыть</button></td></tr>`).join("")}</tbody></table></div><p class="campaign-note">Выбор по этой локальной проверке требует последующего подтверждения на других данных. Баллы площадки здесь не измеряются.</p><details><summary>Основание сравнения</summary><p>Данные <code>${esc(group.dataset_fingerprint)}</code></p><p>Разбиение <code>${esc(group.comparison_key)}</code></p></details></section>`;
  }).join("");
}
function campaignReportGroups(report, campaign) {
  if (report?.campaign?.id !== campaign.id || !Array.isArray(report.groups)) return [];
  return report.groups.map((group) => ({...group, best_job_id: group.best?.job_id,
    control_job_id: group.control?.job_id || group.control_job_id,
    control_value: group.control?.metric?.value ?? group.control_value,
    jobs: (Array.isArray(group.budget_curve) ? group.budget_curve : []).flatMap((point) => {
      const step = (campaign.steps || []).find((step) => step.id === point.step_id && step.job_id === point.job_id);
      return step ? [{job_id: point.job_id, step_id: point.step_id, value: point.metric?.value, action: step.action, minutes: point.minutes,
        elapsed_seconds: point.elapsed_seconds, label: step.label, is_control: step.is_control === true,
        difference: point.difference, improvement: point.improvement}] : [];
    })}));
}
function campaignDetailHTML(campaign, report = null) {
  const steps = Array.isArray(campaign.steps) ? campaign.steps : [], active = campaignRunning(campaign), path = campaignPath(campaign.id);
  const canPause = ["queued", "running"].includes(campaign.status);
  const canResume = ["paused", "interrupted"].includes(campaign.status) && steps.some((step) => step.status === "pending");
  const resumeNote = ["paused", "interrupted"].includes(campaign.status)
    ? `<p class="campaign-note">${campaign.status === "paused" ? "Очередь сохранена за этой серией. Для другого плана остановите эту серию. " : "После прерывания можно явно продолжить оставшиеся задания. "}Продолжение запускает только оставшиеся задания со статусом «Ещё не запускалось»; прежнее обучение не повторяется. Действует первоначальный срок серии${campaign.deadline_at ? `: ${esc(date(campaign.deadline_at))}` : ""}. Пауза и время после закрытия приложения его не продлевают.</p>` : "";
  const controlsBusy = state.campaignControlBusy?.id === campaign.id;
  return `<section class="panel campaign-detail"><div class="panel-head"><div><h2>${esc(campaign.name || "Серия тестов")}</h2><p>${esc(date(campaign.created_at))} · ${campaignCompleted(campaign)} из ${steps.length} заданий выполнены</p></div>${campaignStatus(campaign.status)}</div><div class="panel-body"><div class="campaign-preview-metrics"><div><span>Прошло</span><strong>${duration(campaign.elapsed_seconds)}</strong></div><div><span>Общий лимит</span><strong>${duration(campaign.wall_budget_seconds)}</strong></div><div><span>Потолок исполнений</span><strong>${duration(campaign.worker_upper_bound_seconds)}</strong></div></div><div class="campaign-toolbar">${canPause ? `<button class="button" data-campaign-control="pause" data-campaign-id="${esc(campaign.id)}" ${controlsBusy ? "disabled" : ""}>Пауза после задания</button>` : ""}${canResume ? `<button class="button primary" data-campaign-control="resume" data-campaign-id="${esc(campaign.id)}" ${controlsBusy ? "disabled" : ""}>Продолжить оставшиеся</button>` : ""}${active ? `<button class="button danger" data-campaign-control="stop" data-campaign-id="${esc(campaign.id)}" ${campaign.status === "stopping" || controlsBusy ? "disabled" : ""}>${campaign.status === "cleanup_pending" ? "Повторить остановку" : "Остановить всю серию"}</button>` : ""}</div>${campaign.status === "pausing" ? '<p class="campaign-note">Текущее задание продолжает работу. Следующее будет ждать команды «Продолжить оставшиеся».</p>' : ""}${resumeNote}${campaign.error ? `<div class="warning-box campaign-error">${esc(campaign.error)}</div>` : ""}${campaignSealsHTML(campaign)}<ol class="campaign-step-list campaign-current">${steps.map((step, index) => `<li class="campaign-step"><span class="campaign-step-number">${index + 1}</span><div class="campaign-step-status"><strong>${esc(displayTask(step.task_id))} · ${esc(step.label || CAMPAIGN_ACTION_NAMES[step.action] || step.action)}${step.is_control ? " · контроль" : ""}</strong><p>${["launching", "running"].includes(step.status) ? esc(phaseName(step.job)) : step.job?.metric ? esc(nativeMetric(step.job.metric)) : ""}</p><small>Лимит ${duration(step.worker_timeout_seconds || step.minutes * 60)}${Number.isFinite(step.job?.elapsed_seconds) ? ` · прошло ${duration(step.job.elapsed_seconds)}` : ""}</small>${step.admissible === false ? '<p class="campaign-error">Результат не допущен к сравнению. Изучите эксперимент.</p>' : ""}${step.error ? `<p class="campaign-error">${esc(step.error)}</p>` : ""}</div>${campaignStatus(step.status)}${step.job_id ? `<button class="button small" data-open-job="${esc(step.job_id)}" data-task-id="${esc(step.task_id)}">Эксперимент</button>` : ""}</li>`).join("")}</ol><div class="workbench-section-head"><h3>Результаты серии</h3></div>${report ? campaignComparisonsHTML(campaignReportGroups(report, campaign)) : '<p class="campaign-error">Сводный отчёт пока не загружен. Повторим автоматически; отдельные эксперименты доступны выше.</p>'}<div class="campaign-exports"><a class="button small" href="${path}/report.json" download>${icon("download")} Отчёт JSON</a><a class="button small" href="${path}/report.csv" download>Таблица CSV</a><a class="button small" href="${path}/bundle.zip" download>Пакет для команды ZIP</a></div><p class="campaign-note">Пакет содержит доступные на момент выгрузки отчёты. Локальный лучший результат не равен баллам платформы.</p><details><summary>Версия и ограничения</summary><p>Код <code>${esc(campaign.source_seal?.sha || "Не подтверждён")}</code></p>${warningList(campaign.warnings)}<p>CPU · 8 потоков · API не используется. Остановленные задания не возобновляются с середины.</p></details></div></section>`;
}
async function loadCampaignDetail(id, generation = state.generation) {
  const request = ++state.campaignDetailGeneration;
  const [response, report] = await Promise.all([api(campaignPath(id)), api(`${campaignPath(id)}/report.json`).catch(() => null)]);
  if (generation !== state.generation || state.view !== "campaigns" || state.campaignSelected !== id || request !== state.campaignDetailGeneration) return;
  state.campaignDetail = response.campaign;
  state.campaignReport = report;
  const node = $("#campaign-detail"), opened = $$("details", node).map((detail) => detail.open);
  node.innerHTML = campaignDetailHTML(response.campaign, report);
  $$("details", node).forEach((detail, index) => { detail.open = opened[index] === true; });
}
async function loadCampaignPage(generation) {
  const [presets, response, session, sourceJobs] = await Promise.all([api("/api/campaign-presets"), api("/api/campaigns"), api("/api/session"), api("/api/jobs").catch(() => ({jobs: []}))]);
  if (generation !== state.generation || state.view !== "campaigns") return;
  state.campaignPresets = presets; state.session = session; state.campaigns = Array.isArray(response.campaigns) ? response.campaigns : [];
  setNativeActive(session.active_queue); setCampaignActive(response.active_campaign || session.active_campaign);
  if (!state.campaignDraft) state.campaignDraft = newCampaignDraft();
  state.campaignSourceJobs = Array.isArray(sourceJobs.jobs) ? sourceJobs.jobs : [];
  if (state.campaignStudyRequest) {
    const requested = state.campaignStudyRequest;
    state.campaignStudyRequest = null;
    state.campaignDraft = {...newCampaignDraft(), preset: "lightgbm_trees", task_ids: [requested.task_id], name: "Сравнение обучения LightGBM", minutes_per_step: 30, ...(requested.source_job_id ? {source_job_id: requested.source_job_id} : {})};
    state.campaignDraft.steps = lightgbmStudySteps(state.campaignDraft.task_ids, requested.source_job_id, 30);
    state.campaignEditorOpen = false;
    invalidateCampaignPreview();
  }
  main.innerHTML = heading("ПОСЛЕДОВАТЕЛЬНЫЕ ЭКСПЕРИМЕНТЫ", "Серии тестов", "План на несколько минут или на весь день. Вы задаёте порядок, приложение сохраняет каждый результат.", false) + '<section id="campaign-editor"></section><div class="workbench-section-head"><h2>История и ход работы</h2><span id="campaign-refresh" class="subtle-text">Обновляется каждые 3 секунды</span></div><div class="campaign-layout"><section class="panel campaign-history" id="campaign-history" aria-label="История серий тестов"></section><div id="campaign-detail"></div></div>';
  renderCampaignEditor(); renderCampaignHistory();
  if (!state.campaigns.some((campaign) => campaign.id === state.campaignSelected)) state.campaignSelected = state.campaignActive?.id || state.campaigns[0]?.id || null;
  renderCampaignHistory();
  if (state.campaignSelected) await loadCampaignDetail(state.campaignSelected, generation);
  else $("#campaign-detail").innerHTML = empty("Пока нет запущенных серий", "Проверьте план выше. До нажатия «Запустить серию» обучение не начинается.", "clock", false);
  // Numeric period columns are optional editor choices; failures don't hide the workbench.
  await Promise.allSettled(readyTasks().filter((task) => !state.profiles.has(task.id)).map(async (task) => {
    const profile = await api(`${apiBase(task.id)}/profile`);
    if (generation === state.generation && state.view === "campaigns") state.profiles.set(task.id, profile);
  }));
}
async function previewCampaign() {
  if (state.campaignPreviewBusy) return;
  let body;
  try { body = campaignBody(); } catch (error) { state.campaignError = error.message; renderCampaignEditor(); return; }
  const revision = state.campaignRevision, generation = state.generation;
  state.campaignPreviewBusy = true; state.campaignError = ""; renderCampaignEditor();
  try {
    const preview = await post("/api/campaigns/preview", body);
    if (generation !== state.generation || state.view !== "campaigns" || revision !== state.campaignRevision) return;
    state.campaignPreview = preview; state.campaignPreviewBody = structuredClone(body); state.campaignPreviewRevision = revision;
    // Show the server's exact expansion before the user commits to running it.
    if (Array.isArray(preview.steps) && preview.steps.length) state.campaignDraft.steps = preview.steps.map(({task_id, action, minutes, validation, model_params, model_config, feature_config, is_control, label}) => ({task_id, action, minutes, validation, ...(model_params ? {model_params} : {}), ...(model_config ? {model_config} : {}), ...(feature_config ? {feature_config} : {}), ...(typeof is_control === "boolean" ? {is_control} : {}), ...(label ? {label} : {})}));
  } catch (error) {
    if (generation === state.generation && revision === state.campaignRevision) state.campaignError = error.message;
  } finally { state.campaignPreviewBusy = false; if (generation === state.generation && state.view === "campaigns") renderCampaignEditor(); }
}
async function startCampaign() {
  const block = campaignLaunchBlock(); if (block) { toast(block); return; }
  const body = {...structuredClone(state.campaignPreviewBody), preview_token: state.campaignPreview.plan_sha256}, generation = state.generation;
  state.campaignStartBusy = true; renderCampaignEditor();
  try {
    const response = await post("/api/campaigns", body);
    setCampaignActive(response.campaign); state.campaignSelected = response.campaign.id;
    invalidateCampaignPreview();
    if (generation === state.generation && state.view === "campaigns") await loadCampaignPage(generation);
    else toast("Серия запущена. Ход работы доступен в «Серии тестов».");
  } catch (error) { if (generation === state.generation && state.view === "campaigns") state.campaignError = error.message; else toast(error.message); }
  finally { state.campaignStartBusy = false; if (generation === state.generation && state.view === "campaigns") renderCampaignEditor(); }
}
async function controlCampaign(id, action, button) {
  if (!["pause", "resume", "stop"].includes(action) || state.campaignControlBusy) return;
  state.campaignControlBusy = {id, action}; button.disabled = true;
  try {
    const response = await post(`${campaignPath(id)}/${action}`, {});
    setCampaignActive(campaignRunning(response.campaign) ? response.campaign : null);
    if (state.view === "campaigns" && state.campaignSelected === id) await loadCampaignDetail(id);
    if (state.view === "campaigns") renderCampaignEditor();
  } catch (error) { toast(error.message); }
  finally { state.campaignControlBusy = null; if (button.isConnected) button.disabled = false; }
}
function campaignChanged() {
  state.campaignEditorOpen = true;
  state.campaignDraft.preset = "custom";
  delete state.campaignDraft.source_job_id;
  state.campaignDraft.task_ids = [...new Set(state.campaignDraft.steps.map((step) => step.task_id))];
  invalidateCampaignPreview(); renderCampaignEditor();
}
function selectCampaignPreset(id) {
  const preset = campaignPreset(id); if (!preset) return;
  state.campaignDraft.preset = id;
  state.campaignDraft.name = preset.name;
  state.campaignEditorOpen = id === "custom";
  if (id === "lightgbm_trees") {
    state.campaignDraft.task_ids = [readyTasks().some((task) => task.id === state.task) ? state.task : state.campaignDraft.task_ids[0] || readyTasks()[0]?.id].filter(Boolean);
    state.campaignDraft.minutes_per_step = 30;
    delete state.campaignDraft.source_job_id;
  } else delete state.campaignDraft.source_job_id;
  if (id !== "custom") state.campaignDraft.steps = campaignDefaultSteps(id, state.campaignDraft.task_ids);
  invalidateCampaignPreview(); renderCampaignEditor();
}
function moveCampaignStep(index, direction) {
  const steps = state.campaignDraft?.steps || [], destination = index + direction;
  if (!Number.isInteger(index) || ![-1, 1].includes(direction) || index < 0 || index >= steps.length || destination < 0 || destination >= steps.length) return;
  [steps[index], steps[destination]] = [steps[destination], steps[index]];
  campaignChanged();
}
function changeCampaignField(index, field, value) {
  if (!Number.isInteger(index) || index < 0 || index >= (state.campaignDraft?.steps || []).length) return;
  const step = state.campaignDraft.steps[index];
  if (field === "minutes") step.minutes = Number(value);
  else if (field === "task_id" && readyTasks().some((task) => task.id === value)) {
    const ids = new Set(state.campaignDraft.steps.map((row, position) => position === index ? value : row.task_id));
    if (ids.size > 2) { toast("В одной серии можно использовать не больше двух задач."); renderCampaignEditor(); return; }
    step.task_id = value; step.validation = {strategy: step.action === "automl" ? "auto" : "random"};
  } else if (field === "action" && Object.hasOwn(CAMPAIGN_ACTION_NAMES, value)) {
    step.action = value; if (value !== "automl") step.minutes = 30;
    if (value !== "automl" && step.validation?.strategy === "auto") step.validation = {strategy: "random"};
    delete step.model_params; delete step.model_config; delete step.feature_config;
  } else if (field === "search_strategy" && step.action === "automl" && ["adaptive", "legacy"].includes(value)) {
    step.search_strategy = value;
  } else if (field === "strategy" && campaignValidationOptions(step).some(([strategy]) => strategy === value)) {
    step.validation = {strategy: value};
  } else if (field === "time_column") step.validation = {strategy: "time", time_column: value};
  else return;
  campaignChanged();
}
async function pollWorkbench() {
  if (document.hidden || !["overview", "campaigns"].includes(state.view) || state.overviewPollBusy || state.campaignPollBusy) return;
  const generation = state.generation;
  if (state.view === "overview") {
    state.overviewPollBusy = true;
    try { await loadOverview(generation, true); }
    catch { if (generation === state.generation && $("#overview-refresh")) $("#overview-refresh").textContent = "Связь прервалась. Повторим автоматически."; }
    finally { state.overviewPollBusy = false; }
    return;
  }
  state.campaignPollBusy = true;
  try {
    const [response, session] = await Promise.all([api("/api/campaigns"), api("/api/session")]);
    if (generation !== state.generation || state.view !== "campaigns") return;
    state.session = session; state.campaigns = Array.isArray(response.campaigns) ? response.campaigns : [];
    setNativeActive(session.active_queue); setCampaignActive(response.active_campaign || session.active_campaign);
    renderCampaignHistory();
    if (state.campaignSelected) await loadCampaignDetail(state.campaignSelected, generation);
    const block = campaignLaunchBlock(), start = $('[data-action="campaign-start"]');
    if (start) start.disabled = Boolean(block);
    if ($("#campaign-launch-block")) $("#campaign-launch-block").textContent = block || "План проверен. Можно запускать.";
    if ($("#campaign-refresh")) $("#campaign-refresh").textContent = `Обновлено ${new Date().toLocaleTimeString("ru-RU")}`;
  } catch { if (generation === state.generation && $("#campaign-refresh")) $("#campaign-refresh").textContent = "Связь прервалась. Повторим автоматически."; }
  finally { state.campaignPollBusy = false; }
}

// Native series use the pinned agent and retain their own history and audits.
const NATIVE_PROFILES = ["regression_off", "regression_on", "classification"];
const NATIVE_NAMES = {
  regression_off: "Регрессия · rescue OFF",
  regression_on: "Регрессия · rescue ON",
  classification: "Классификация",
};
const NATIVE_STATUS = {
  queued: ["В очереди", "gray"], pending: ["Ожидает", "gray"],
  preparing: ["Подготовка", "amber"], running: ["Выполняется", "red"],
  stopping: ["Останавливается", "amber"], completed: ["Завершено", "green"],
  failed: ["Ошибка", "red"], stopped: ["Очередь остановлена", "amber"],
  interrupted: ["Прервано", "amber"], cancelled: ["Остановлено", "gray"],
  cleanup_pending: ["Остановка требует проверки", "amber"],
  not_started: ["Не запускалось", "gray"],
};
function nativeStatus(status) {
  const [label, tone] = NATIVE_STATUS[status] || ["Статус не подтверждён", "gray"];
  return badge(label, tone);
}
function nativeRunning(queue) {
  return Boolean(queue && ["queued", "preparing", "running", "stopping", "cleanup_pending"].includes(queue.status));
}
function nativePath(id) {
  if (typeof id !== "string" || !/^[0-9a-f]{32}$/.test(id)) throw new Error("Неверный идентификатор серии.");
  return `/api/native-queues/${encodeURIComponent(id)}`;
}
function nativeTasks(type) {
  return state.tasks.filter((task) => task.available === true && task.task_type === type);
}
function nativeMetric(metric) {
  return Number.isFinite(metric?.value) ? `${number(metric.value, 6)} ${metric.name || ""}`.trim() : "Пока нет метрики";
}
function nativePayload(regressionTask, classificationTask, seconds) {
  if (!nativeTasks("regression").some((task) => task.id === regressionTask) ||
      !nativeTasks("classification").some((task) => task.id === classificationTask))
    throw new Error("Выберите готовые задачи регрессии и классификации.");
  if (!Number.isInteger(seconds) || seconds < 301 || seconds > 10500)
    throw new Error("Лимит одного задания — целое число секунд от 301 до 10 500.");
  return {regression_task_id: regressionTask, classification_task_id: classificationTask, time_budget_seconds: seconds};
}
function nativeLaunchBlock() {
  if (state.nativeStartBusy) return "Серия запускается…";
  if (campaignRunning(state.campaignActive)) return "Серия тестов занимает очередь. Продолжите или остановите её в разделе «Серии тестов».";
  if (nativeRunning(state.nativeActiveQueue)) return "Сначала дождитесь завершения текущей серии или остановите её.";
  if (runningJob(state.session?.active_job)) return "Сейчас выполняется другой запуск приложения.";
  if (!state.nativePreflight) return "Сначала проверьте готовность компьютера.";
  if (!state.nativePreflight.ready) return "Перед запуском устраните причины, указанные в проверке готовности.";
  if (!nativeTasks("regression").length || !nativeTasks("classification").length)
    return "Нужны готовые задачи регрессии и классификации.";
  return null;
}
function setNativeActive(queue) {
  state.nativeActiveQueue = nativeRunning(queue) ? queue : null;
  state.nativeActivityUpdated = Date.now();
  applyNativeLaneControls();
}
function applyNativeLaneControls() {
  const blocked = nativeRunning(state.nativeActiveQueue) || campaignRunning(state.campaignActive);
  $$('[data-launch]').forEach((button) => {
    if (blocked && button.dataset.nativeBlocked !== "true") {
      button.dataset.nativeWasDisabled = String(button.disabled);
      button.dataset.nativeBlocked = "true";
      button.disabled = true;
    } else if (!blocked && button.dataset.nativeBlocked === "true") {
      button.disabled = button.dataset.nativeWasDisabled === "true";
      delete button.dataset.nativeBlocked;
      delete button.dataset.nativeWasDisabled;
    }
  });
}
function nativeOptions(type, selected) {
  const tasks = nativeTasks(type);
  return tasks.length ? tasks.map((task) => `<option value="${esc(task.id)}" ${task.id === selected ? "selected" : ""}>${esc(task.name || task.id)}</option>`).join("")
    : '<option value="">Нет готовой задачи</option>';
}
function nativePreflightPanel() {
  const report = state.nativePreflight;
  const errors = Array.isArray(report?.errors) ? report.errors : [];
  return `<div class="native-readiness-head"><div><h3>Готовность компьютера</h3><p>Одна серия занимает общую очередь CPU. Нужны Windows, Python 3.12 и 32 ГБ памяти.</p></div><button class="button small" data-action="native-preflight" ${state.nativePreflightBusy ? "disabled" : ""}>${state.nativePreflightBusy ? '<span class="spinner"></span> Проверяем…' : `${icon("shield")} Проверить готовность`}</button></div>${report ? `<div class="native-readiness-result">${badge(report.ready ? "Готов к запуску" : "Запуск пока недоступен", report.ready ? "green" : "amber")}<span>CPU: ${number(report.threads || 8)} потоков</span>${Number.isFinite(report.physical_ram_bytes) ? `<span>RAM: ${number(report.physical_ram_bytes / 1024 ** 3, 1)} ГиБ</span>` : ""}${Number.isFinite(report.free_disk_bytes) ? `<span>Свободно: ${number(report.free_disk_bytes / 1024 ** 3, 1)} ГиБ</span>` : ""}</div>${errors.length ? warningList(errors.map((error) => typeof error === "string" ? error : error?.message || error?.reason || "Проверка не завершена")) : ""}${report.release_sha ? `<p class="native-source">Версия агента <code>${esc(report.release_sha)}</code></p>` : ""}` : '<p class="native-hint">Проверка подтвердит среду, свободное место и доступность CPU.</p>'}`;
}
function nativeShell() {
  const block = nativeLaunchBlock();
  return heading("ПРОВЕРКА ВЕРСИИ АГЕНТА", "Проверка агента", "Специальный протокол: сравнение rescue OFF / ON и классификация с проверкой отчётов. Для обычного обучения используйте «Серии тестов».", false) +
    `<section class="panel native-plan"><div class="native-plan-title"><div><h2>Новая серия</h2><p>Каждое задание получит отдельную копию данных и собственный отчёт.</p></div>${badge("Без API", "green")}</div><ol class="native-plan-steps">${NATIVE_PROFILES.map((profile, index) => `<li><span class="native-step-number">${index + 1}</span><div><strong>${esc(NATIVE_NAMES[profile])}</strong><small>${index === 0 ? "Контрольный результат" : index === 1 ? "Та же задача и те же входные данные" : "Отдельная проверка классификации"}</small></div></li>`).join("")}</ol><form id="native-launch-form"><div class="form-grid"><label class="form-field">Регрессия (требуется RMSE)<select id="native-regression-task" name="regression_task_id" ${nativeTasks("regression").length ? "" : "disabled"} required>${nativeOptions("regression", state.nativeRegressionTask)}</select></label><label class="form-field">Классификация<select id="native-classification-task" name="classification_task_id" ${nativeTasks("classification").length ? "" : "disabled"} required>${nativeOptions("classification", state.nativeClassificationTask)}</select></label><label class="form-field">Лимит одного задания<select id="native-budget" name="time_budget_seconds">${[[3600, "До 1 часа"], [10500, "До 2 ч 55 мин"]].map(([value, label]) => `<option value="${value}" ${value === state.nativeBudgetSeconds ? "selected" : ""}>${label}</option>`).join("")}</select></label><div class="native-budget-summary"><span>Потолок обучения всей серии</span><strong id="native-total-budget">${duration(state.nativeBudgetSeconds * 3)}</strong><small>Подготовка и проверка отчётов займут дополнительное время.</small></div></div><p class="native-hint">Поиск по бюджету: до 256 попыток, с резервом на финальное обучение. Это верхние лимиты: агент может закончить раньше. Серия работает на CPU в 8 потоков. Видеокарта в этом плане не используется.</p><div id="native-readiness" class="native-readiness">${nativePreflightPanel()}</div><p id="native-launch-error" class="form-error" role="alert"></p><div class="native-launch-footer"><p id="native-launch-block">${esc(block || "Задания выполняются по одному. Ошибка обязательной проверки остановит очередь.")}</p><button class="button primary" id="native-start" type="submit" ${block ? "disabled" : ""}>${icon("activity")} ${state.nativeStartBusy ? "Запускаем…" : "Запустить три задания"}</button></div></form></section>` +
    `<div class="native-history-heading"><h2>История серий</h2><span id="native-refresh-time" class="subtle-text">Обновление каждые 3 секунды</span></div><div class="native-layout"><section class="panel native-history-panel" aria-label="История ночных тестов"><div id="native-history" class="native-history"></div></section><div class="native-detail-column"><section id="native-queue-detail"></section><section class="panel native-logs" id="native-log-panel" hidden><div class="panel-head"><div><h2>Журнал серии</h2><p id="native-log-caption">Журнал координатора · обновляется во время работы</p></div></div><div class="native-log-controls"><label class="form-field">Источник<select id="native-log-kind"><option value="coordinator">Координатор · текущий журнал</option><option value="stdout">stdout · итог задания</option><option value="stderr">stderr · итог задания</option></select></label><label class="form-field" id="native-log-profile-field" hidden>Задание<select id="native-log-profile">${NATIVE_PROFILES.map((profile) => `<option value="${profile}">${esc(NATIVE_NAMES[profile])}</option>`).join("")}</select></label></div><pre id="native-log" class="code-content native-log-text" tabindex="0">Выберите серию.</pre></section></div></div>`;
}
function nativeCheckLabel(name) {
  return ({root_exit_zero: "Процесс завершился успешно", watchdog_not_reached: "Внешний лимит не превышен",
    memory_limit_not_reached: "Лимит памяти не превышен", tree_resources_observed: "Ресурсы процесса измерены",
    owned_cleanup_complete: "Дочерние процессы завершены", input_hashes_unchanged: "Входные данные не изменились",
    flat_hashes_unchanged: "Код агента не изменился", worker_config_unchanged: "Настройки задания не изменились",
    source_identity_unchanged: "Версия исходников сохранена", worker_returned: "Получен отчёт текущего запуска",
    observer_complete: "Наблюдения сохранены", one_adaptive_offline_invocation: "Выполнен offline-поиск",
    original_deadline_respected: "Исходный бюджет соблюдён", zero_api: "API не использовался",
    fresh_full_refit_published: "Есть новое обучение на полном TRAIN", selected_outer_execution_clean: "Проверочное обучение завершено",
    selected_refit_execution_clean: "Полное обучение завершено", requested_rescue_profile_recorded: "Режим rescue подтверждён",
    output_structure_valid: "Формат предсказаний корректен", outer_and_full_refit_receipts_valid: "Квитанции обучения проверены"})[name] || name.replaceAll("_", " ");
}
function nativeProfileCard(row) {
  const resources = row.resources || {};
  const measured = resources.available === true;
  const stage = ({inner_research: "Поиск на INNER", outer_validation: "Проверка на outer", full_refit: "Обучение на полном TRAIN",
    preparing: "Подготовка", startup: "Подготовка исходной модели", baseline: "Подготовка исходной модели",
    waiting: "Ожидает очереди", not_started: "Следующее задание не запускалось",
    result: "Результат", finished: "Завершение"})[row.stage] || row.stage;
  const checks = row.checks && typeof row.checks === "object" ? Object.entries(row.checks) : [];
  return `<article class="native-profile-card ${row.status === "running" ? "active" : ""}"><div class="native-profile-head"><h3>${esc(NATIVE_NAMES[row.profile] || row.profile)}</h3>${nativeStatus(row.status)}</div><div class="native-profile-metric">${esc(nativeMetric(row.metric))}</div><p class="native-profile-stage">${esc(stage || "Обучение ещё не началось")}${Number.isFinite(row.elapsed_seconds) ? ` · ${duration(row.elapsed_seconds)}` : ""}</p><p class="native-resource">${measured ? `Измеренный пик RAM: ${Number.isFinite(resources.peak_rss_bytes) ? `${number(resources.peak_rss_bytes / 1024 ** 3, 2)} ГиБ` : "не указан"}` : "Измерения ресурсов пока не подтверждены"}</p>${row.continuation_allowed === true ? `<p class="native-admission good">${icon("check")} Обязательные проверки пройдены</p>` : row.continuation_allowed === false ? `<p class="native-admission bad">${icon("missing")} Переход к следующему заданию запрещён</p>` : ""}${row.qualification === "requires_review" ? '<p class="native-review">Результат требует разбора. Это отдельно от технического допуска продолжения.</p>' : ""}${row.error ? `<p class="native-error">${esc(row.error)}</p>` : ""}${checks.length ? `<details class="native-audit" data-profile="${esc(row.profile)}"><summary>Проверки завершения · ${checks.filter(([, value]) => value === true).length}/${checks.length}</summary><ul>${checks.map(([name, value]) => `<li><span>${esc(nativeCheckLabel(name))}</span>${badge(value === true ? "Пройдено" : value === false ? "Не пройдено" : "Неизвестно", value === true ? "green" : value === false ? "red" : "gray")}</li>`).join("")}</ul></details>` : ""}</article>`;
}
function nativeComparison(comparison) {
  if (!comparison || comparison.status === "incomplete") return '<div class="native-comparison"><h3>Сравнение OFF / ON</h3><p>Появится после двух успешно проверенных прогонов регрессии.</p></div>';
  if (comparison.status !== "comparable") return '<div class="native-comparison"><h3>Сравнение OFF / ON</h3><p>Результаты не подтверждены как совместимые. Изучите отчёты каждого задания.</p></div>';
  return `<div class="native-comparison"><h3>Сравнение OFF / ON</h3><div class="native-comparison-values"><span>OFF<strong>${esc(nativeMetric(comparison.off))}</strong></span><span>ON<strong>${esc(nativeMetric(comparison.on))}</strong></span><span>RMSE · ON − OFF<strong>${Number.isFinite(comparison.on_minus_off) ? number(comparison.on_minus_off, 6) : "—"}</strong></span></div><p>Отрицательная разница RMSE означает меньшую ошибку ON в этой проверке.</p>${comparison.rescue_exercised === false ? badge("Rescue был пропущен", "amber") : comparison.rescue_exercised === true ? badge("Rescue был запущен", "gray") : badge("Выполнение rescue не подтверждено", "gray")}<p>Одна пара на известных данных не доказывает общий прирост качества.</p></div>`;
}
function nativeDetailHTML(queue) {
  const rows = NATIVE_PROFILES.map((profile) => {
    const row = (queue.profiles || []).find((item) => item.profile === profile) || {profile, status: "pending"};
    return row.status === "pending" && ["stopped", "cancelled", "interrupted", "failed", "cleanup_pending"].includes(queue.status)
      ? {...row, status: "not_started", stage: "not_started"} : row;
  });
  const stopLabel = queue.status === "cleanup_pending" ? "Повторить остановку" : "Остановить всю очередь";
  return `<section class="panel native-detail"><div class="panel-head"><div><h2>Серия ${esc(queue.id.slice(0, 8))}</h2><p>${esc(date(queue.created_at))} · до ${duration(queue.total_native_seconds || queue.time_budget_seconds * 3)} обучения</p></div>${nativeStatus(queue.status)}</div><div class="panel-body"><div class="native-series-actions">${nativeRunning(queue) ? `<button class="button danger" data-action="native-stop" data-native-id="${esc(queue.id)}" ${queue.status === "stopping" ? "disabled" : ""}>${icon("missing")} ${stopLabel}</button>` : ""}${queue.artifact_available ? `<a class="button" href="${nativePath(queue.id)}/evidence" download>${icon("download")} Скачать отчёты ZIP</a>` : '<span class="subtle-text">Архив появится после сохранения отчётов. При прерывании часть файлов может быть доступна только в папке результатов.</span>'}</div>${queue.error ? `<div class="warning-box native-error">${esc(queue.error)}</div>` : ""}<p class="native-hint">${queue.research_search === "adaptive-budget" ? "Поиск по бюджету · до 256 попыток" : "Прежний поиск · до 64 попыток"}</p><div class="native-profile-grid">${rows.map(nativeProfileCard).join("")}</div>${nativeComparison(queue.comparison)}<p class="native-hint">Метрики рассчитаны на известных проверочных строках TRAIN и не равны баллам платформы.</p><details class="native-location"><summary>Версия и папка результатов</summary><p>Версия агента <code>${esc(queue.release_sha || "Не подтверждена")}</code></p><p>Результаты <code>${esc(queue.output_dir || "Папка ещё не назначена")}</code></p><p>Прерванные серии сохраняются в истории. Продолжение старого обучения автоматически не запускается.</p></details></div></section>`;
}
function renderNativeHistory() {
  const node = $("#native-history");
  if (!node) return;
  node.innerHTML = state.nativeQueues.length ? state.nativeQueues.map((queue) => `<button class="native-history-item ${queue.id === state.nativeSelectedQueue ? "selected" : ""}" data-native-queue="${esc(queue.id)}" aria-pressed="${queue.id === state.nativeSelectedQueue}"><span><strong>Серия ${esc(queue.id.slice(0, 8))}</strong><small>${esc(date(queue.created_at))}</small></span>${nativeStatus(queue.status)}</button>`).join("") : '<div class="job-empty"><h3>Серий пока нет</h3><p>Выберите две задачи и проверьте готовность компьютера.</p></div>';
}
function updateNativeLaunchState() {
  const block = nativeLaunchBlock(), start = $("#native-start");
  if (start) {
    start.disabled = Boolean(block);
    start.innerHTML = `${icon("activity")} ${state.nativeStartBusy ? "Запускаем…" : "Запустить три задания"}`;
  }
  const note = $("#native-launch-block");
  if (note) note.textContent = block || "Задания выполняются по одному. Ошибка обязательной проверки остановит очередь.";
}
async function loadNativePage(generation) {
  const [history, tasks, session, preflight] = await Promise.all([
    api("/api/native-queues"), api("/api/tasks"), api("/api/session"),
    state.nativePreflight ? Promise.resolve(state.nativePreflight) : api("/api/native-queues/preflight").catch((error) => ({ready: false, errors: [error.message]})),
  ]);
  if (generation !== state.generation || state.view !== "native") return;
  state.tasks = Array.isArray(tasks.tasks) ? tasks.tasks : [];
  if (!state.tasks.some((task) => task.id === state.task)) state.task = state.tasks.find((task) => task.available)?.id || state.tasks[0]?.id || null;
  const taskSelect = $("#task-select");
  taskSelect.innerHTML = state.tasks.length ? state.tasks.map((task) => `<option value="${esc(task.id)}">${esc(taskName(task))}</option>`).join("") : "<option>Задач пока нет</option>";
  taskSelect.value = state.task || "";
  taskSelect.disabled = !state.tasks.length;
  state.session = session;
  setCampaignActive(session.active_campaign);
  state.nativePreflight = preflight;
  state.nativeQueues = Array.isArray(history.queues) ? history.queues : [];
  setNativeActive(history.active_queue);
  if (!nativeTasks("regression").some((task) => task.id === state.nativeRegressionTask)) state.nativeRegressionTask = nativeTasks("regression")[0]?.id || null;
  if (!nativeTasks("classification").some((task) => task.id === state.nativeClassificationTask)) state.nativeClassificationTask = nativeTasks("classification")[0]?.id || null;
  if (!state.nativeQueues.some((queue) => queue.id === state.nativeSelectedQueue)) state.nativeSelectedQueue = state.nativeActiveQueue?.id || state.nativeQueues[0]?.id || null;
  main.innerHTML = nativeShell();
  renderNativeHistory();
  if (state.nativeSelectedQueue) await loadNativeDetail(state.nativeSelectedQueue, generation);
  else $("#native-queue-detail").innerHTML = empty("Готово к ночным тестам", "После запуска здесь появятся этапы, метрики и проверки каждого задания.", "clock", false);
}
async function checkNativePreflight() {
  if (state.nativePreflightBusy) return;
  state.nativePreflightBusy = true;
  const generation = state.generation;
  $("#native-readiness").innerHTML = nativePreflightPanel();
  try {
    const [value, session] = await Promise.all([api("/api/native-queues/preflight"), api("/api/session")]);
    if (generation === state.generation && state.view === "native") {
      state.nativePreflight = value;
      state.session = session;
    }
  } catch (error) {
    if (generation === state.generation) state.nativePreflight = {ready: false, errors: [error.message]};
  } finally {
    state.nativePreflightBusy = false;
    if (generation === state.generation && state.view === "native") {
      $("#native-readiness").innerHTML = nativePreflightPanel();
      updateNativeLaunchState();
    }
  }
}
async function loadNativeDetail(id, generation = state.generation) {
  const request = ++state.nativeDetailGeneration;
  const response = await api(nativePath(id));
  if (generation !== state.generation || state.view !== "native" || state.nativeSelectedQueue !== id || request !== state.nativeDetailGeneration) return;
  const queue = response.queue;
  state.nativeDetail = queue;
  const open = new Set($$(".native-audit[open]").map((node) => node.dataset.profile));
  $("#native-queue-detail").innerHTML = nativeDetailHTML(queue);
  $$(".native-audit").forEach((node) => { node.open = open.has(node.dataset.profile); });
  $("#native-log-panel").hidden = false;
  $("#native-log-kind").value = state.nativeLogKind;
  $("#native-log-profile").value = state.nativeLogProfile;
  $("#native-log-profile-field").hidden = state.nativeLogKind === "coordinator";
  await loadNativeLog(id, generation);
}
async function loadNativeLog(id = state.nativeSelectedQueue, generation = state.generation) {
  if (!id) return;
  const kind = state.nativeLogKind, profile = state.nativeLogProfile, request = ++state.nativeLogGeneration;
  const query = new URLSearchParams({kind});
  if (kind !== "coordinator") query.set("profile", profile);
  try {
    const response = await api(`${nativePath(id)}/logs?${query}`);
    if (generation !== state.generation || state.view !== "native" || state.nativeSelectedQueue !== id || kind !== state.nativeLogKind || profile !== state.nativeLogProfile || request !== state.nativeLogGeneration) return;
    const node = $("#native-log"), atEnd = node.scrollTop + node.clientHeight >= node.scrollHeight - 10;
    node.textContent = response.text || (kind === "coordinator" ? "В журнале пока нет записей." : "Итоговый журнал появится после завершения задания.");
    if (atEnd) node.scrollTop = node.scrollHeight;
    $("#native-log-caption").textContent = kind === "coordinator" ? "Журнал координатора · обновляется во время работы" : `${kind} · ${NATIVE_NAMES[profile]} · ${response.finalized ? "сохранённый итоговый журнал" : "будет сохранён после завершения задания"}`;
  } catch (error) {
    if (generation === state.generation && state.view === "native" && state.nativeSelectedQueue === id && request === state.nativeLogGeneration)
      $("#native-log").textContent = error.message;
  }
}
async function startNativeQueue(event) {
  event.preventDefault();
  if (nativeLaunchBlock()) return;
  const generation = state.generation, errorNode = $("#native-launch-error");
  errorNode.textContent = "";
  let payload;
  try { payload = nativePayload(state.nativeRegressionTask, state.nativeClassificationTask, state.nativeBudgetSeconds); }
  catch (error) { errorNode.textContent = error.message; return; }
  state.nativeStartBusy = true;
  updateNativeLaunchState();
  try {
    const response = await post("/api/native-queues", payload);
    setNativeActive(response.queue);
    state.nativeSelectedQueue = response.queue.id;
    if (generation === state.generation && state.view === "native") await loadNativePage(generation);
    toast("Серия запущена. Задания выполняются последовательно.");
  } catch (error) {
    if (generation === state.generation && state.view === "native") errorNode.textContent = error.message;
    else toast(error.message);
  } finally {
    state.nativeStartBusy = false;
    if (state.view === "native") updateNativeLaunchState();
  }
}
async function stopNativeQueue(id, button) {
  button.disabled = true;
  try {
    const response = await post(`${nativePath(id)}/stop`, {});
    if (!nativeRunning(response.queue)) setNativeActive(null);
    if (state.view === "native" && state.nativeSelectedQueue === id) await loadNativeDetail(id);
    toast("Остановка всей очереди запрошена.");
  } catch (error) { toast(error.message); }
  finally { if (button.isConnected) button.disabled = false; }
}
async function pollNativeQueues() {
  if (document.hidden || state.view !== "native" || state.nativePollBusy) return;
  const generation = state.generation;
  state.nativePollBusy = true;
  try {
    const [response, session] = await Promise.all([api("/api/native-queues"), api("/api/session")]);
    if (generation !== state.generation || state.view !== "native") return;
    state.session = session;
    setCampaignActive(session.active_campaign);
    state.nativeQueues = Array.isArray(response.queues) ? response.queues : [];
    setNativeActive(response.active_queue);
    renderNativeHistory();
    updateNativeLaunchState();
    if (state.nativeSelectedQueue) await loadNativeDetail(state.nativeSelectedQueue, generation);
    $("#native-refresh-time").textContent = `Обновлено ${new Date().toLocaleTimeString("ru-RU")}`;
  } catch {
    if (generation === state.generation && $("#native-refresh-time")) $("#native-refresh-time").textContent = "Нет обновления. Сохранённые результаты остаются на экране.";
  } finally { state.nativePollBusy = false; }
}
async function syncNativeActivity() {
  if (state.nativeActivityBusy || Date.now() - state.nativeActivityUpdated < 3000) return;
  state.nativeActivityBusy = true;
  try {
    const session = await api("/api/session");
    state.session = session;
    setNativeActive(session.active_queue);
    setCampaignActive(session.active_campaign);
  }
  catch { /* The server remains authoritative if activity cannot be refreshed. */ }
  finally { state.nativeActivityBusy = false; }
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
    const [data, session, native] = await Promise.all([
      api("/api/tasks"),
      api("/api/session"),
      api("/api/native-queues").catch(() => ({active_queue: null})),
    ]);
    if (generation === state.generation) state.session = session;
    if (generation !== state.generation) return;
    setNativeActive(native.active_queue);
    setCampaignActive(session.active_campaign);
    state.tasks = Array.isArray(data.tasks) ? data.tasks : [];
    const select = $("#task-select");
    if (!state.tasks.length) {
      state.task = null;
      select.innerHTML = "<option>Задач пока нет</option>";
      select.disabled = true;
      if (["overview", "campaigns", "native"].includes(state.view)) {
        await renderView();
        return;
      }
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
          `<option value="${esc(t.id)}">${esc(taskName(t))}${t.available ? "" : " · нет данных"}</option>`,
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
  const names = {overview: "Обзор", data: "Данные", runs: "Эксперименты", result: "Результаты", campaigns: "Серии тестов", native: "Проверка агента"};
  $$(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === state.view);
    if (button.dataset.view === state.view)
      button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("#crumb-name").textContent = names[state.view];
  if (["overview", "campaigns"].includes(state.view)) {
    main.innerHTML = loading(state.view === "overview" ? "Открываем обзор лаборатории…" : "Открываем серии тестов…");
    try {
      if (state.view === "overview") await loadOverview(generation);
      else await loadCampaignPage(generation);
    } catch (error) {
      if (generation === state.generation) main.innerHTML = errorView(error, "Не удалось открыть раздел");
    }
    return;
  }
  if (state.view === "native") {
    main.innerHTML = loading("Открываем серии тестов…");
    try {
      await loadNativePage(generation);
    } catch (error) {
      if (generation === state.generation) main.innerHTML = errorView(error, "Не удалось открыть проверку агента");
    }
    return;
  }
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
    applyNativeLaneControls();
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
  cleanup_pending: ["Остановка требует проверки", "amber"],
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
const RESEARCH_STAGE_NAMES = {
  preparation: "Подготовка",
  inner_research: "Внутренние эксперименты",
  candidate_frozen: "Фиксация кандидата",
  outer_validation: "Внешняя проверка",
  full_refit: "Обучение на всех данных",
  result: "Результат",
};
const RESEARCH_REASON_NAMES = {
  api_error: "Ошибка обращения к API",
  time_limit: "Достигнут лимит времени",
  step_limit: "Достигнут лимит шагов",
  truncated_response: "Ответ модели обрезан",
  host_metric_unsupported: "Для этой метрики проверка качества не поддерживается",
  repeated_failures: "Повторяющиеся ошибки экспериментов",
  context_budget: "Достигнут лимит контекста модели",
  inner_validation_unavailable: "Не удалось сформировать внутреннюю проверку",
  refit_failed: "Не удалось завершить обучение на всех данных",
};
function researchFallbackReason(reason) {
  return Object.hasOwn(RESEARCH_REASON_NAMES, reason)
    ? RESEARCH_REASON_NAMES[reason] : "Причина завершения не указана";
}
function researchProvenance(summary, imported = false) {
  if (!summary || summary.protocol_version !== 3)
    return '<span class="research-origin legacy">Прежний протокол v2 · разделение данных не подтверждено</span>';
  return `<span class="research-origin">Протокол v3${imported ? " · отчёт участника, локальная проверка не выполнялась" : " · внутренний подбор и внешняя приёмка"}</span>`;
}
function researchMetric(metric) {
  return metric && Number.isFinite(metric.value)
    ? `<strong>${esc(metric.name)} ${number(metric.value, 6)}</strong><span>${metric.higher_is_better ? "Больше лучше" : "Меньше лучше"}</span>`
    : '<span class="subtle-text">Оценка ещё не получена</span>';
}
function researchSplit(split) {
  if (!split) return "";
  const strategy = { random: "Случайное разбиение", time: "По времени", grouped: "По группам дублей" }[split.strategy] || "";
  const rows = Number.isFinite(split.train_rows) && Number.isFinite(split.validation_rows)
    ? `Обучение ${number(split.train_rows)} / проверка ${number(split.validation_rows)}` : "";
  return [strategy, rows].filter(Boolean).join(" · ");
}
function researchView(summary, { status, reason, imported = false } = {}) {
  const provenance = researchProvenance(summary, imported);
  const fallback = status === "fallback"
    ? `<p class="research-fallback">Сохранён резервный результат. ${esc(researchFallbackReason(reason))}. Исследование Qwen не завершилось штатно.</p>` : "";
  if (!summary || summary.protocol_version !== 3)
    return `<section class="research-summary legacy" aria-label="Происхождение исследования">${provenance}${fallback}</section>`;
  const stages = Array.isArray(summary.stages) ? summary.stages : [];
  const terminal = ["completed", "fallback", "failed", "cancelled", "interrupted"].includes(status);
  const labels = { pending: terminal ? "Не выполнялся" : "Ожидает", running: "Выполняется", completed: "Готово", skipped: "Пропущен", failed: "Ошибка" };
  const selection = summary.evaluations?.selection;
  const check = summary.evaluations?.final_check;
  const frozen = summary.frozen_candidate;
  return `<section class="research-summary" aria-label="Ход исследования Qwen">${provenance}${fallback}<ol class="research-stages">${Object.entries(RESEARCH_STAGE_NAMES).map(([id, name]) => {
    const item = stages.find((entry) => entry.id === id) || { status: "pending" };
    const stageStatus = Object.hasOwn(labels, item.status) ? item.status : "pending";
    const active = stageStatus === "running" && !terminal;
    const explanation = item.reason ? (Object.hasOwn(RESEARCH_REASON_NAMES, item.reason) ? RESEARCH_REASON_NAMES[item.reason] : "")
      || ({ skipped: "Условия этапа не выполнены", failed: "Подробности в журнале запуска" }[stageStatus]) : "";
    return `<li class="research-stage ${stageStatus}"${summary.stage === id ? ' aria-current="step"' : ""}><span class="research-stage-name">${esc(name)}</span><span class="research-stage-status">${stageStatus === "completed" ? icon("check") : active ? '<span class="spinner"></span>' : ""}${stageStatus === "running" && terminal ? "Остановлен" : labels[stageStatus]}</span>${explanation ? `<small>${esc(explanation)}</small>` : ""}</li>`;
  }).join("")}</ol><div class="research-evaluations"><div class="research-score"><h3>Внутренний подбор</h3>${researchMetric(selection?.metric || (selection?.value != null ? selection : null))}<small>${esc(researchSplit(selection?.split))}</small></div><div class="research-score"><h3>Внешняя приёмка</h3>${check ? `<div class="research-outer-scores"><div><span>Стартовый рецепт</span>${researchMetric(check.baseline)}</div><div><span>Кандидат</span>${researchMetric(check.candidate)}</div></div>${check.selected ? `<p>${check.selected === "candidate" ? "Выбран кандидат" : "Сохранён стартовый рецепт"}</p>` : ""}<small>${esc(researchSplit(check.split))}</small>` : '<span class="subtle-text">Проводится после фиксации рецепта</span>'}</div></div><p class="research-score-note">Внешняя выборка используется для приёмки. Оценка выбранного результата не является независимой оценкой качества.</p>${frozen?.code_hash ? `<details class="research-hashes"><summary>Рецепт зафиксирован</summary><p>Код <code>${esc(frozen.code_hash.slice(0, 12))}</code>${frozen.recipe_hash ? ` · параметры <code>${esc(frozen.recipe_hash.slice(0, 12))}</code>` : ""}</p></details>` : ""}</section>`;
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
      if (result.selection?.verified && !result.error && result.exit_code === 0
          && !result.timed_out && !result.memory_limit_exceeded) {
        const selected = result.selection;
        const metric = selected.metric;
        color = "success";
        body = selected.reason === "not_better"
          ? "Эксперимент проверен, улучшения нет. Предыдущий результат сохранён."
          : selected.accepted ? "Эксперимент проверен и принят." : "Эксперимент проверен.";
        if (metric && Number.isFinite(metric.value))
          body += ` ${metric.name}: ${number(metric.value, 6)}.`;
      }
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
  const trial = result.selection || {};
  const rawFamily = trial.recipe_summary?.family || trial.recipe?.family || trial.family;
  const family = typeof rawFamily === "string" && /^[\p{L}\p{N} ._+()-]{1,120}$/u.test(rawFamily)
    ? rawFamily : rawFamily ? "Пользовательский рецепт" : "";
  const scope = trial.scope === "inner_selection" ? "Внутренняя проверка · подбор рецепта" : "";
  const evidence = [scope, trial.hypothesis, family,
    Number.isFinite(result.elapsed_seconds) ? duration(result.elapsed_seconds) : ""].filter((item) => typeof item === "string" && item);
  return `<div class="timeline-event ${color}"><span class="event-dot"></span><div class="event-header"><span class="event-title">${esc(title)}</span><span class="event-index">${String(index + 1).padStart(2, "0")}</span></div><p class="event-body">${esc(body)}</p>${evidence.length ? `<p class="event-details">${evidence.map(esc).join(" · ")}</p>` : ""}${result.output_check?.errors?.length ? `<p class="event-details">${result.output_check.errors.map(esc).join(" · ")}</p>` : ""}</div>`;
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
      host_metric_unsupported: "Качество требует отдельной проверки",
      repeated_failures: "Повторяющиеся ошибки экспериментов",
    };
    $("#run-detail").innerHTML =
      `<div class="run-detail-header"><div><h2>Запуск ${esc(id.slice(0, 8))}</h2><p>${esc(date(r.started_at))} · ${esc(r.model || "Модель не указана")}</p><div class="refresh-time"><span class="connection-dot"></span><span id="last-refresh">Обновление каждые 4 секунды</span></div></div>${statusBadge(r.status)}</div><div class="stats-grid">${stat("Время запуска", duration(r.elapsed_seconds), "по последнему отчёту", "clock", true)}${stat("Исполнений", number(r.experiments), `${number(r.successful_experiments)} с корректным CSV`, "code")}${stat("Запросов API", number(detail.usage?.attempts ?? r.api?.attempts), "попыток обращения", "activity")}${stat("Формат CSV", r.output_check?.valid === true ? "Корректен" : r.output_check?.valid === false ? "Ошибка" : "Не проверен", "не оценка ML-качества", "check", true)}</div><div class="usage-row">${usageView(detail.usage)}</div>${warningList(detail.warnings)}${researchView(detail.research || { protocol_version: r.research_protocol_version || 2, stage: r.stage, stages: r.stages, evaluations: r.evaluations, frozen_candidate: r.frozen_candidate }, { status: r.status, reason: r.reason })}<section class="panel"><div class="panel-head"><div><h2>Журнал событий</h2><p>События в порядке выполнения</p></div>${badge(`${number(r.events?.length || 0)} событий`, "gray")}</div><div class="timeline">${r.events?.length ? r.events.map(eventView).join("") : '<p class="event-body">События ещё не записаны.</p>'}</div></section><section class="panel"><div class="panel-head"><div><h2>Код и вывод</h2><p>Файлы выбранного эксперимента</p></div>${experiments.length ? `<label class="sr-only" for="experiment-select">Выбрать эксперимент</label><select id="experiment-select" class="file-select">${experiments.map((value) => `<option value="${value}"${state.experiment === value ? " selected" : ""}>${value === 0 ? "Стартовая модель" : `Эксперимент ${value}`}</option>`).join("")}</select>` : ""}</div><div class="code-tabs" role="tablist" aria-label="Файл эксперимента">${[
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
function comparisonGroups(reports) {
  const groups = new Map();
  for (const report of reports) {
    if (report.source !== "local" || report.status !== "completed" ||
      report.task_id !== state.task ||
      !state.currentDatasetFingerprint || !report.dataset_fingerprint ||
      report.dataset_fingerprint !== state.currentDatasetFingerprint ||
      !report.comparison_key || !report.metric ||
      !Number.isFinite(report.metric.value) ||
      typeof report.metric.higher_is_better !== "boolean") continue;
    const key = [report.dataset_fingerprint, report.comparison_key,
      report.metric.name, report.metric.higher_is_better].join("|");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(report);
  }
  return [...groups].map(([key, reports]) => {
    const rows = [...reports].sort((a, b) => String(b.started_at || "").localeCompare(String(a.started_at || "")) || String(b.id).localeCompare(String(a.id)));
    const best = rows.reduce((a, b) => (b.metric.higher_is_better
      ? b.metric.value > a.metric.value : b.metric.value < a.metric.value) ? b : a);
    return { key, rows, best, latest: rows[0].started_at || "" };
  }).sort((a, b) => b.latest.localeCompare(a.latest) || a.key.localeCompare(b.key));
}
function selectedSubmissionJob(reports) {
  if (!state.currentDatasetFingerprint) return null;
  const eligible = reports.filter((job) => job.source === "local" &&
    job.status === "completed" && job.task_id === state.task &&
    job.dataset_fingerprint === state.currentDatasetFingerprint && job.output_valid === true);
  const selected = state.workflow?.published_job_id;
  return eligible.find((job) => job.id === selected && job.published !== false) ||
    eligible.find((job) => job.published === true) || null;
}
function resultGroupsPanel(reports) {
  const groups = comparisonGroups(reports);
  return `<section class="result-quality" id="result-groups"><div class="result-section-head"><div><h2>Качество по схемам проверки</h2><p>У каждой схемы свой лучший результат. Оценки между схемами не сравниваются.</p></div><button class="button" data-launch="baseline">Новый эксперимент</button></div>${groups.length ? `<div class="quality-group-grid">${groups.map(({rows,best}) => `<article class="panel quality-group-card"><div class="panel-head"><div><h3>${esc(validationLabel(best))}</h3><p>${rows.length} ${rows.length === 1 ? "результат" : rows.length < 5 ? "результата" : "результатов"} на одних проверочных строках</p></div>${badge(rows.length > 1 ? "Лучший в этой проверке" : "Первый результат", "gray")}</div><div class="panel-body"><div class="metric-main"><strong class="metric-number">${number(best.metric.value,6)}</strong><span class="metric-name">${esc(best.metric.name)}</span></div><p class="direction-note">${icon(best.metric.higher_is_better ? "arrowup" : "arrowdown")}${best.metric.higher_is_better ? "Больше — лучше" : "Меньше — лучше"}</p><p class="group-model">${esc(jobTitle(best))} · ${esc(best.model || "")}</p><p class="validation-evidence">${esc(validationEvidence(best))}${Number.isFinite(best.split?.validation_fraction) ? ` · проверка ${number(best.split.validation_fraction*100,1)} % строк` : ""}</p>${modelParameterView(best,true)}<div class="group-actions"><button class="button small" data-open-job="${esc(best.id)}" data-task-id="${esc(state.task)}">Открыть запуск</button>${isFixedRecipe(best.action) ? `<button class="button small" data-edit-job="${esc(best.id)}">Изменить настройки</button>` : ""}${best.published ? badge("Этот CSV выбран", "green") : best.output_valid === true ? `<button class="button small" data-action="publish-job" data-publish-id="${esc(best.id)}">Использовать CSV</button>` : ""}</div></div></article>`).join("")}</div>` : `<div class="panel result-no-quality"><div class="quality-empty">${icon("chart")}<div><h3>Измерьте качество в приложении</h3><p>После обучения здесь появится результат выбранной схемы проверки.</p><button class="button primary" data-launch="baseline">Обучить baseline</button></div></div></div>`}</section>`;
}
function submissionPanel(data, reports) {
  const submission = data?.submission || {},
    selected = submission.valid === true ? selectedSubmissionJob(reports) : null;
  const title = !submission.exists ? "Файл ещё не создан" : submission.valid === true
    ? "Формат проверен" : submission.valid === false ? "Есть ошибки в CSV" : "Проверка недоступна";
  return `<section class="panel current-submission" id="current-submission"><div class="panel-head"><div><h2>Текущий CSV</h2><p>Выбранный файл предсказаний · submission.csv</p></div>${selected ? badge("Выбран из приложения", "green") : badge("Отдельный файл", "gray")}</div><div class="panel-body submission-content"><div><div class="result-state"><span class="result-state-icon ${submission.valid === false ? "bad" : submission.valid !== true ? "warn" : ""}">${icon(submission.valid === true ? "check" : "missing")}</span><div><h3>${title}</h3><p>${submission.valid === true ? "Столбцы, значения и порядок ID корректны." : "Подготовьте или выберите предсказания завершённого запуска."}</p></div></div>${selected ? `<div class="selected-csv-source"><span class="selected-csv-caption">Выбранный запуск</span><button class="text-button" data-open-job="${esc(selected.id)}" data-task-id="${esc(state.task)}">${esc(jobTitle(selected))} · ${esc(date(selected.started_at))}</button>${validationView(selected,true)}${Number.isFinite(selected.metric?.value) ? `<p>Его локальная метрика: <strong>${number(selected.metric.value,6)} ${esc(selected.metric.name)}</strong></p>` : ""}</div>` : `<p class="run-note">${submission.exists ? "Этот файл найден в папке задачи. Его связь с запуском приложения не подтверждена." : "Обучите модель, затем нажмите «Использовать CSV» у нужного запуска."}</p>`}</div><div><div class="file-box">${icon("file")}<div><strong>submission.csv</strong><small>${esc(submission.path || "Папка выбранной задачи")}</small>${submission.exists && Number.isFinite(submission.size_bytes) ? `<small>${number(submission.size_bytes / 1024,1)} КБ</small>` : ""}</div></div>${submission.exists ? `<a class="button ${submission.valid === true ? "primary" : ""}" href="${apiBase()}/submission" download>${icon("download")} Скачать текущий CSV</a>` : '<button class="button" data-launch="baseline">Обучить модель</button>'}${warningList(submission.errors)}<p class="run-note">Выбор CSV выполняется отдельно. Проверка формата не измеряет качество модели.</p></div></div></section>`;
}
function legacyBaselinePanel(data) {
  const baseline = data?.baseline, metric = baseline?.metric, split = baseline?.split || {};
  return `<details class="panel legacy-result" id="legacy-result"><summary><span>Прежний отчёт из терминала</span><small>baseline_metrics.json · отдельно от запусков приложения</small></summary><div class="panel-body">${baseline ? `<div class="legacy-result-value"><strong>${Number.isFinite(metric?.value) ? number(metric.value,6) : "—"}</strong><span>${esc(metric?.name || "Метрика не указана")}</span></div><p>${esc(baseline.model?.name || "Модель не указана")}</p>${validationView({source:"local",action:"baseline",split},true)}<p class="run-note">Этот отчёт не определяет качество текущего CSV и не участвует в выборе лучшего запуска приложения.</p>` : `<p class="run-note">${esc(data?.baseline_error || "Отчёта из терминала пока нет.")}</p>`}</div></details>`;
}
function resultFingerprint(data, response) {
  return JSON.stringify([response.dataset_fingerprint,response.experiments,data,state.workflow?.published_job_id]);
}
function command(text) {
  return `<div class="command-row"><code>${esc(text)}</code><button data-copy="${esc(text)}" aria-label="Скопировать команду" title="Скопировать">${icon("copy")}</button></div>`;
}
function showHelp() {
  const apiInfo = state.session?.api || {};
  $("#help-title").textContent = "Настройка среды";
  $("#help-content").innerHTML =
    `<div class="steps"><div class="help-step"><span>1</span><div class="help-step-content"><h3>Запуски на этом компьютере</h3><p>Baseline и ExtraTrees можно обучать кнопками в панели. Результаты каждого эксперимента сохраняются отдельно.</p></div></div><div class="help-step"><span>2</span><div class="help-step-content"><h3>Подключение Qwen</h3><p>${apiInfo.ready ? "Настройки API заполнены. Подключение проверяется при запуске исследования." : "Заполните настройки организаторов в локальном .env и перезапустите сервер."}</p>${!apiInfo.ready ? `<div class="command-row"><code>${esc((apiInfo.missing || ["OPENAI_API_KEY", "OPENAI_BASE_URL"]).join("\n"))}</code></div>` : ""}</div></div><div class="help-step"><span>3</span><div class="help-step-content"><h3>Работа команды</h3><p>В разделе «Результат» можно скачать отчёт JSON или ZIP-пакет запуска и добавить его в панель коллеги. Импорт добавляет только отчёт; текущий CSV выбирается отдельно.</p></div></div></div><p class="help-note">Каждый участник запускает панель на своём компьютере. Общего онлайн-сервера в этой версии нет.</p>`;
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
  if (!["overview", "data", "runs", "result", "campaigns", "native"].includes(view)) return;
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
  if (button.dataset.lightgbmStudySource) openLightgbmStudy(button.dataset.lightgbmStudySource, button.dataset.taskId);
  else if (button.dataset.action === "lightgbm-study") openLightgbmStudy();
  else if (button.dataset.action === "training-device-probe") refreshTrainingDevices();
  else if (button.dataset.campaignPreset) selectCampaignPreset(button.dataset.campaignPreset);
  else if (button.dataset.campaignMove !== undefined) moveCampaignStep(Number(button.dataset.campaignMove), Number(button.dataset.direction));
  else if (button.dataset.campaignRemove !== undefined) {
    const index = Number(button.dataset.campaignRemove);
    if (Number.isInteger(index) && index >= 0 && index < state.campaignDraft.steps.length) {
      state.campaignDraft.steps.splice(index, 1); campaignChanged();
    }
  } else if (button.dataset.action === "campaign-add") {
    const draft = state.campaignDraft, task_id = draft.task_ids[0] || readyTasks()[0]?.id;
    if (draft.steps.length >= 12 || !task_id) return;
    draft.steps.push({task_id, action: "automl", minutes: 30, validation: {strategy: "auto"}}); campaignChanged();
  } else if (button.dataset.action === "campaign-preview") previewCampaign();
  else if (button.dataset.action === "campaign-start") startCampaign();
  else if (button.dataset.campaignControl) controlCampaign(button.dataset.campaignId, button.dataset.campaignControl, button);
  else if (button.dataset.campaignOpen) {
    state.campaignSelected = button.dataset.campaignOpen;
    if (state.view !== "campaigns") navigate("campaigns");
    else { renderCampaignHistory(); loadCampaignDetail(button.dataset.campaignOpen).catch((error) => toast(error.message)); }
  } else if (button.dataset.overviewTask) {
    state.task = button.dataset.overviewTask; $("#task-select").value = state.task; state.workflow = null; navigate("data");
  } else if (button.dataset.nativeQueue) {
    state.nativeSelectedQueue = button.dataset.nativeQueue;
    renderNativeHistory();
    loadNativeDetail(button.dataset.nativeQueue).catch((error) => toast(error.message));
  } else if (button.dataset.action === "native-preflight") checkNativePreflight();
  else if (button.dataset.action === "native-stop") stopNativeQueue(button.dataset.nativeId, button);
  else if (button.dataset.closeDialog) $("#" + button.dataset.closeDialog)?.close();
  else if (button.dataset.launch) showLaunch(button.dataset.launch);
  else if (button.dataset.editJob) editJobSettings(button.dataset.editJob);
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
setInterval(pollNativeQueues, 3000);
setInterval(pollWorkbench, 3000);
document.addEventListener("change", (event) => {
  const target = event.target;
  if (target.dataset.campaignIndex !== undefined) changeCampaignField(Number(target.dataset.campaignIndex), target.dataset.campaignField, target.value);
  else if (target.dataset.campaignTask !== undefined) {
    const id = target.dataset.campaignTask, draft = state.campaignDraft;
    const ids = draft.preset === "lightgbm_trees" ? [id] : target.checked ? [...draft.task_ids.filter((item) => item !== id), id] : draft.task_ids.filter((item) => item !== id);
    if (draft.preset === "lightgbm_trees" && !draft.task_ids.includes(id)) delete draft.source_job_id;
    if (ids.length > 2) { target.checked = false; toast("Выберите не больше двух задач для одной серии."); return; }
    draft.task_ids = ids;
    draft.steps = draft.preset === "custom" ? draft.steps.filter((step) => ids.includes(step.task_id)) : campaignDefaultSteps(draft.preset, ids);
    if (draft.preset === "custom") for (const task_id of ids) if (!draft.steps.some((step) => step.task_id === task_id) && draft.steps.length < 12) draft.steps.push({task_id, action: "automl", minutes: 30, validation: {strategy: "auto"}});
    invalidateCampaignPreview(); renderCampaignEditor();
  } else if (target.id === "campaign-source-job" || target.id === "campaign-minutes-per-step") {
    const draft = state.campaignDraft;
    if (draft?.preset !== "lightgbm_trees") return;
    updateLightgbmStudy(target.id === "campaign-source-job" ? target.value : draft.source_job_id,
      target.id === "campaign-minutes-per-step" ? target.value : draft.minutes_per_step);
  } else if (target.id === "campaign-name") { state.campaignDraft.name = target.value; invalidateCampaignPreview(); renderCampaignEditor(); }
  else if (target.id === "campaign-max-hours") { state.campaignDraft.max_hours = Number(target.value); invalidateCampaignPreview(); renderCampaignEditor(); }
  else if (event.target.id === "native-regression-task") state.nativeRegressionTask = event.target.value;
  else if (event.target.id === "native-classification-task") state.nativeClassificationTask = event.target.value;
  else if (event.target.id === "native-budget") {
    state.nativeBudgetSeconds = Number(event.target.value);
    $("#native-total-budget").textContent = duration(state.nativeBudgetSeconds * 3);
  } else if (event.target.id === "native-log-kind") {
    if (!["coordinator", "stdout", "stderr"].includes(event.target.value)) return;
    state.nativeLogKind = event.target.value;
    $("#native-log-profile-field").hidden = state.nativeLogKind === "coordinator";
    loadNativeLog();
  } else if (event.target.id === "native-log-profile") {
    if (!NATIVE_PROFILES.includes(event.target.value)) return;
    state.nativeLogProfile = event.target.value;
    loadNativeLog();
  }
});
document.addEventListener("submit", (event) => {
  if (event.target.id === "native-launch-form") startNativeQueue(event);
});
document.addEventListener("toggle", (event) => {
  if (event.target.id === "campaign-step-editor") state.campaignEditorOpen = event.target.open;
}, true);
if (["overview", "data", "runs", "result", "campaigns", "native"].includes(location.hash.slice(1)))
  state.view = location.hash.slice(1);
loadTasks();

// Controlled jobs and team reports. The server owns commands and file access.
const ACTION_NAMES = {
  automl: "Автоподбор",
  baseline: "Baseline",
  extra_trees: "ExtraTrees",
  catboost: "CatBoost",
  lightgbm: "LightGBM",
  linear: "Линейная модель",
  agent: "Qwen",
  prepare: "Загрузка данных",
};
const FIXED_RECIPES = ["baseline", "extra_trees", "catboost", "lightgbm", "linear"];
const isFixedRecipe = (action) => FIXED_RECIPES.includes(action);
const isOfflineRecipe = (action) => action === "automl" || isFixedRecipe(action);
function launchRecipeOptions(profile) {
  return [
    ["automl", "Автоподбор", "Поиск моделей и признаков · без API"],
    ...FIXED_RECIPES.filter((recipe) => modelSchema(profile, recipe)).map((recipe) => [
      recipe, ACTION_NAMES[recipe], recipe === "baseline" ? "Стартовая точка · без API" : `${modelSchema(profile, recipe).model} · без API`,
    ]),
    ["agent", "Qwen", "Пишет код · общий API-бюджет"],
  ];
}
const STAGE_NAMES = {
  preparing: "Подготовка данных",
  training: "Обучение модели",
  validating: "Проверка на holdout",
  refitting: "Обучение на всей выборке",
  saving_model: "Сохранение обученной модели",
  finished: "Работа завершена",
  research: "Исследование с Qwen",
  downloading: "Загрузка публичных данных",
  cleanup: "Остановка требует проверки",
};
const runningJob = (job) => job && ["queued", "running", "cleanup_pending"].includes(job.status);
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
          : "Запустите автоподбор",
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
  applyNativeLaneControls();
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
      : "Подберите модель за один запуск";
  const description = measured
    ? `${latest.metric.name}: ${number(latest.metric.value, 6)} · ${latest.metric.higher_is_better ? "больше лучше" : "меньше лучше"}`
    : "Сначала получите результат, затем сравните модели и признаки. Автоподбор работает без API.";
  const primary = measured
    ? firstComparison
      ? '<button class="button primary" data-launch="automl">Запустить автоподбор</button>'
      : '<button class="button primary" data-workflow-view="result">Сравнить результаты</button>'
    : `<button class="button primary" data-launch="automl">${icon("search")} Автоподбор</button>`;
  return `<section class="next-action"><div><h2>${esc(title)}</h2><p>${esc(description)}</p></div><div class="action-buttons">${primary}${measured && !firstComparison ? '<button class="button" data-launch="automl">Новый автоподбор</button>' : '<button class="button" data-launch="baseline">Одна модель</button>'}<button class="button" data-launch="agent">${ready ? "Исследовать с Qwen" : "Подключить Qwen"} ${icon("activity")}</button></div></section>`;
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
  const [jobsResult, profileResult] = await Promise.allSettled([
    api(`/api/jobs${task ? "?task_id=" + encodeURIComponent(task) : ""}`),
    task && !modelSchema(currentProfile(), "baseline")
      ? api(`${apiBase(task)}/profile`) : Promise.resolve(null),
  ]);
  if (generation !== state.generation || task !== state.task) return;
  if (profileResult.status === "fulfilled" && profileResult.value)
    state.profiles.set(task, profileResult.value);
  if (jobsResult.status === "rejected") throw jobsResult.reason;
  const data = jobsResult.value;
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
      `<div class="run-start-actions"><button class="button primary" data-launch="automl" ${state.task ? "" : "disabled"}>${icon("search")} Автоподбор</button><button class="button" data-launch="baseline" ${state.task ? "" : "disabled"}>Одна модель</button><button class="button" data-launch="extra_trees" ${state.task ? "" : "disabled"}>ExtraTrees</button><button class="button" data-action="lightgbm-study" ${state.task ? "" : "disabled"}>${icon("test")} Сравнить обучение LightGBM</button><span class="subtle-text">Без API · на этом компьютере</span></div><div class="jobs-layout" style="margin-top:20px"><section class="panel run-list-panel"><div class="panel-head"><h2>История</h2><span class="subtle-text" id="job-count"></span></div><div id="job-list" class="job-list"></div></section><section id="job-detail"></section></div>`;
    mountWorkflow();
  }
  renderJobList();
  if (state.selectedJob) await loadJobDetail(state.selectedJob, quiet);
  else
    $("#job-detail").innerHTML =
      `<section class="panel"><div class="job-empty"><p>Автоподбор сравнит варианты и сохранит лучший CSV. Здесь появится ход работы.</p><button class="button primary" data-launch="${state.task ? "automl" : "prepare"}">${state.task ? "Запустить автоподбор" : "Загрузить данные"}</button></div></section>`;
}
function renderJobList() {
  if (!$("#job-list")) return;
  $("#job-count").textContent = String(state.jobs.length);
  $("#job-list").innerHTML =
    state.jobs
      .map(
        (job) =>
          `<button class="job-item ${job.id === state.selectedJob ? "selected" : ""}" data-job="${esc(job.id)}" aria-pressed="${job.id === state.selectedJob}"><h3>${esc(jobTitle(job))}</h3>${statusBadge(job.agent_status === "fallback" ? "fallback" : job.status)}<p>${esc(job.author || "Без автора")} · ${duration(job.elapsed_seconds)}</p>${isOfflineRecipe(job.action) ? `<p class="job-validation-label">${esc(validationLabel(job))}</p><p class="job-model-label">${esc(job.action === "automl" && !job.model_config ? "Поиск моделей и признаков" : modelSettingsState(job).label)}</p>` : ""}<p>${esc(date(job.started_at))}</p></button>`,
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
function trainedModelView(job) {
  if (job?.source !== "local" || job.status !== "completed" || job.output_valid !== true) return "";
  const artifact = job.model_artifact;
  if (!artifact) return "";
  if (artifact.available !== true) return `<section class="trained-model-panel"><h3>Обученная модель</h3><p>${esc(artifact.reason || "Для этого запуска отдельный файл обученной модели не сохранён.")}</p>${lightgbmStudySource(job, job.task_id) ? '<p>Настройки победителя можно использовать кнопкой «Сравнить обучение LightGBM».</p>' : ""}</section>`;
  if (!/^[0-9a-f]{32}$/.test(job.id || "") || !/^[0-9a-f]{64}$/.test(artifact.sha256 || "") ||
      !Number.isFinite(artifact.size_bytes) || artifact.size_bytes <= 0 || artifact.format !== "joblib") return "";
  const base = `/api/jobs/${encodeURIComponent(job.id)}`, query = `?task_id=${encodeURIComponent(job.task_id)}`;
  return `<section class="trained-model-panel" aria-label="Сохранённая обученная модель"><div class="training-artifact-heading"><div><h3>Обученная модель сохранена</h3><p>Финальное обучение на полном train · ${number(artifact.size_bytes / 1024 ** 2, 1)} МиБ</p></div>${badge("Локальная модель", "green")}</div><p>Файл содержит обученную модель и обработку признаков. Метрика выше получена на отдельном обучении с отложенными строками.</p><div class="action-buttons"><a class="button" href="${base}/model${query}" download>${icon("download")} Модель .joblib</a><a class="button" href="${base}/model-manifest${query}" download>Паспорт модели JSON</a><a class="button" href="${base}/recipe${query}" download>Рецепт обучения JSON</a></div><p class="campaign-note">Это модель для локального использования. Кнопка «Собрать solution.zip» выгружает код конкурсного агента отдельно.</p><details><summary>Проверка файла</summary><p>SHA-256 <code>${esc(artifact.sha256)}</code></p></details></section>`;
}
function trainingDeviceEvidenceView(job) {
  if (job?.source !== "local" || job.training_device !== "gpu") return "";
  const evidence = job.training_device_evidence;
  return `<div class="training-device-result">${evidence?.fit_completed === true && evidence.backend_verified === true && evidence.backend === "GPU"
    ? `${badge("Обучено на GPU", "green")}<span>${esc(evidence.device_name || "GPU · CatBoost")}</span>`
    : `${badge("GPU запрошен", "amber")}<span>Завершённое обучение на GPU пока не подтверждено.</span>`}<p>Экспериментальное GPU-обучение. Время и метрика относятся к этому устройству; воспроизводимость на CPU проверяется отдельно.</p></div>`;
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
  if (job?.action === "automl" && job?.validation?.strategy === "auto")
    return "Автоматический выбор проверки";
  return job?.source === "local" &&
    isFixedRecipe(job?.action) &&
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
function currentProfile() {
  const profile = state.profiles.get(state.task) ||
    (state.profile?.id === state.task ? state.profile : null);
  const fingerprint = state.workflow?.dataset_fingerprint || state.currentDatasetFingerprint;
  return profile?.dataset_fingerprint && fingerprint &&
    profile.dataset_fingerprint !== fingerprint ? null : profile;
}
function modelSchema(profile, action) {
  const schema = profile?.model_options?.[action];
  return schema && typeof schema.model === "string" &&
    schema.defaults && typeof schema.defaults === "object" &&
    Array.isArray(schema.fields) ? schema : null;
}
function modelSettingsState(job) {
  const config = job?.model_config;
  if (!config?.parameters || typeof config.parameters !== "object")
    return { kind: "legacy", changed: 0, label: "Параметры не сохранены" };
  if (job.action === "automl")
    return { kind: "saved", changed: 0, label: "Параметры победителя" };
  const schema = modelSchema(currentProfile(), job.action);
  if (!schema || schema.model !== config.model)
    return { kind: "saved", changed: 0, label: "Параметры сохранены" };
  const keys = new Set([...Object.keys(schema.defaults), ...Object.keys(config.parameters)]);
  const changed = [...keys].filter(
    (key) => !Object.hasOwn(schema.defaults, key) ||
      !Object.hasOwn(config.parameters, key) ||
      JSON.stringify(schema.defaults[key]) !== JSON.stringify(config.parameters[key]),
  ).length;
  return changed
    ? { kind: "changed", changed, label: `Изменено: ${number(changed)}` }
    : { kind: "standard", changed: 0, label: "Стандартные настройки" };
}
function isStandardBaseline(job) {
  return job?.action === "baseline" &&
    (!job.model_config || modelSettingsState(job).kind === "standard");
}
function jobTitle(job) {
  const title = ACTION_NAMES[job?.action] || job?.model || "Эксперимент";
  const settings = modelSettingsState(job);
  return isFixedRecipe(job?.action) && settings.kind === "changed"
    ? `${title} · свои настройки` : title;
}
function displayModelValue(field, value) {
  const option = field?.options?.find((item) => JSON.stringify(item.value) === JSON.stringify(value));
  if (option) return option.label;
  if (value && typeof value === "object") return JSON.stringify(value);
  return value === null ? "Без ограничения" : String(value);
}
function modelParameterView(job, compact = false) {
  const settings = modelSettingsState(job),
    parameters = job?.model_config?.parameters;
  if (!parameters || typeof parameters !== "object")
    return `<span class="model-settings-note">${esc(settings.label)}</span>`;
  const profile = currentProfile(),
    schema = modelSchema(profile, job.action) || (job.action === "automl"
      ? Object.values(profile?.model_options || {}).find((item) => item.model === job.model_config.model)
      : null),
    fields = schema?.fields || [];
  return `<details class="model-parameters${compact ? " compact" : ""}"><summary>${esc(settings.label)}</summary><dl>${Object.entries(parameters).slice(0, 64).map(([name, value]) => {
    const field = fields.find((item) => item.name === name);
    if (["FixedBlend", "WeightedBlend"].includes(job.model_config.model) && name === "components") return `<dt>Модели ансамбля</dt><dd>${value.map((item) => esc(item.model)).join(" + ")}</dd>`;
    if (job.model_config.model === "WeightedBlend" && name === "weights") return `<dt>Веса моделей</dt><dd>${value.map((weight) => `${number(weight * 100, 1)} %`).join(" / ")}</dd>`;
    return `<dt>${esc(field?.label || name)}</dt><dd>${esc(displayModelValue(field, value))}</dd>`;
  }).join("")}</dl></details>`;
}
function parseModelParams(schema, values) {
  const parameters = {};
  for (const field of schema.fields) {
    const raw = values[field.name];
    if (typeof raw !== "string") throw new Error(`Заполните поле «${field.label}».`);
    if (field.type === "select") {
      const option = (field.options || []).find((item) => JSON.stringify(item.value) === raw);
      if (!option) throw new Error(`Выберите значение поля «${field.label}».`);
      parameters[field.name] = option.value;
      continue;
    }
    if (!raw.trim()) {
      if (field.nullable) { parameters[field.name] = null; continue; }
      throw new Error(`Заполните поле «${field.label}».`);
    }
    const value = Number(raw);
    if (!Number.isFinite(value) ||
      (field.type === "integer" && !Number.isInteger(value)) ||
      (field.min != null && value < field.min) ||
      (field.max != null && value > field.max))
      throw new Error(`Проверьте значение поля «${field.label}».`);
    parameters[field.name] = value;
  }
  return parameters;
}
function modelDraft(schema, config = null) {
  const parameters = config?.model === schema.model && config.parameters
    ? config.parameters : schema.defaults;
  return Object.fromEntries(schema.fields.map((field) => {
    const value = Object.hasOwn(parameters, field.name)
      ? parameters[field.name] : schema.defaults[field.name];
    return [field.name, field.type === "select" ? JSON.stringify(value) : value == null ? "" : String(value)];
  }));
}
function readModelDraft() {
  return Object.fromEntries($$("[data-model-field]", $("#launch-form")).map(
    (field) => [field.dataset.modelField, field.value],
  ));
}
function saveModelDraft() {
  if (modelSchema(state.launchProfile, state.launchModelRecipe))
    state.launchModelDrafts[state.launchModelRecipe] = readModelDraft();
}
function updateModelSummary() {
  const schema = modelSchema(state.launchProfile, state.launchModelRecipe),
    summary = $("#model-changes");
  if (!schema || !summary) return;
  const values = readModelDraft(), defaults = modelDraft(schema),
    changed = schema.fields.filter((field) => values[field.name] !== defaults[field.name]).length;
  summary.textContent = changed ? `Изменено: ${number(changed)}` : "Стандартные настройки";
  summary.classList.toggle("changed", changed > 0);
}
function mountModelSettings(action) {
  const container = $("#model-settings"), schema = modelSchema(state.launchProfile, action);
  state.launchModelRecipe = action;
  if (!container) return;
  if (!schema) {
    container.innerHTML = '<summary><span>Настройки модели</span><span class="subtle-text">Стандартный рецепт</span></summary><p class="model-schema-note">Для этой версии доступен стандартный рецепт.</p>';
    return;
  }
  const values = state.launchModelDrafts[action] || modelDraft(schema);
  state.launchModelDrafts[action] = values;
  container.innerHTML = `<summary><span>Настройки модели</span><span id="model-changes" class="model-changes"></span></summary><fieldset id="model-fields"><div class="model-settings-head"><span>${esc(schema.model)}</span><button class="text-button" type="button" id="model-reset">Сбросить</button></div><div class="model-field-grid">${schema.fields.map((field) => `<label class="form-field">${esc(field.label)}${field.type === "select"
    ? `<select data-model-field="${esc(field.name)}">${(field.options || []).map((option) => `<option value="${esc(JSON.stringify(option.value))}" ${values[field.name] === JSON.stringify(option.value) ? "selected" : ""}>${esc(option.label)}</option>`).join("")}</select>`
    : `<input type="number" data-model-field="${esc(field.name)}" value="${esc(values[field.name] || "")}" ${field.min == null ? "" : `min="${esc(field.min)}"`} ${field.max == null ? "" : `max="${esc(field.max)}"`} step="${field.type === "integer" ? "1" : "any"}" ${field.nullable ? 'placeholder="Без ограничения"' : "required"}>`}</label>`).join("")}</div></fieldset>`;
  $("#model-reset").onclick = () => {
    state.launchModelDrafts[action] = modelDraft(schema);
    mountModelSettings(action);
    container.open = true;
  };
  updateModelSummary();
}
function launchPayload(values) {
  const { task_id, action, author, hypothesis, minutes } = values,
    payload = { task_id, action, author, hypothesis, minutes };
  if (isOfflineRecipe(action)) {
    if (values.validation) payload.validation = values.validation;
    if (action === "automl") payload.search_strategy = values.search_strategy || "adaptive";
    if (action !== "automl") {
      if (values.model_config) payload.model_config = values.model_config;
      else if (values.model_params) payload.model_params = values.model_params;
      if (values.feature_config) payload.feature_config = values.feature_config;
      if (action === "catboost" && values.training_device) payload.training_device = values.training_device;
    }
  }
  return payload;
}
function editJobSettings(id) {
  const job = state.jobDetail?.job?.id === id ? state.jobDetail.job :
    [...state.jobs, ...state.experimentReports].find((item) => item.id === id);
  if (!currentLocalResult(job) || !isFixedRecipe(job.action)) {
    toast("Выберите локальный завершённый запуск на текущих данных.");
    return;
  }
  showLaunch(job.action, job);
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
    earlier.find(isStandardBaseline) ||
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
        isStandardBaseline(job) ? "Стартовая точка" : "Первая оценка модели",
      note: "Первая оценка на этой схеме проверки.",
      tone: "neutral",
      symbol: "test",
    };
  const delta = job.metric.value - reference.metric.value;
  const improvement = job.metric.higher_is_better ? delta : -delta;
  const equal = Math.abs(delta) < 1e-9;
  const referenceName =
    isStandardBaseline(reference) ? "стандартного baseline" : "лучшего прежнего запуска";
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
  return `<section class="job-metric">${outcome ? `<div class="job-outcome ${outcome.tone}"><span class="job-outcome-icon">${icon(outcome.symbol)}</span><div><h3>${esc(outcome.title)}</h3><p>${esc(outcome.note)}</p></div></div>` : ""}<div class="job-metric-value"><strong>${number(job.metric.value, 6)}</strong><span>${esc(job.metric.name)}</span></div><p>${esc(job.model || ACTION_NAMES[job.action])} · проверка${job.split?.validation_fraction != null ? " на " + number(job.split.validation_fraction * 100, 1) + " % данных" : " на отложенных данных"} · ${job.metric.higher_is_better ? "больше лучше" : "меньше лучше"}</p>${validationView(job)}${modelParameterView(job)}${next ? `<div class="job-next-step">${next}</div>` : ""}</section>${classificationDiagnosticView(job)}`;
}
function classificationDiagnosticEvidence(job) {
  const data = job?.classification_diagnostics;
  if (job?.status !== "completed" || job.metric?.name !== "Macro F1" ||
      job.metric?.higher_is_better !== true || !Number.isFinite(job.metric?.value) ||
      !data || data.version !== 1 || data.kind !== "classification" ||
      !Array.isArray(data.labels) || !data.labels.length || data.labels.length > 50 ||
      data.labels.some((label) => typeof label !== "string" || !label.trim() || Array.from(label).length > 200) ||
      new Set(data.labels).size !== data.labels.length ||
      !Number.isSafeInteger(data.rows) || data.rows <= 0 ||
      !Array.isArray(data.confusion_matrix) || data.confusion_matrix.length !== data.labels.length ||
      !Array.isArray(data.per_class) || data.per_class.length !== data.labels.length) return null;
  const size = data.labels.length,
    count = (value) => Number.isSafeInteger(value) && value >= 0 && value <= 2147483647,
    unit = (value) => Number.isFinite(value) && value >= 0 && value <= 1,
    close = (a, b) => Number.isFinite(a) && Math.abs(a - b) <= 1e-10;
  if (data.confusion_matrix.some((row) => !Array.isArray(row) || row.length !== size || row.some((value) => !count(value)))) return null;
  const supports = data.confusion_matrix.map((row) => row.reduce((sum, value) => sum + value, 0)),
    predicted = data.labels.map((_, index) => data.confusion_matrix.reduce((sum, row) => sum + row[index], 0));
  if (supports.reduce((sum, value) => sum + value, 0) !== data.rows ||
      (Number.isInteger(job.split?.validation_rows) && job.split.validation_rows !== data.rows)) return null;
  const rows = data.labels.map((label, index) => {
    const correct = data.confusion_matrix[index][index], support = supports[index], total = predicted[index],
      precision = total ? correct / total : 0, recall = support ? correct / support : 0,
      f1 = support + total ? 2 * correct / (support + total) : 0;
    return { label, index, support, predicted: total, precision, recall, f1 };
  });
  if (rows.some((row, index) => {
    const saved = data.per_class[index];
    return !saved || saved.support !== row.support || saved.predicted !== row.predicted ||
      ["precision", "recall", "f1"].some((key) => !unit(saved[key]) || !close(saved[key], row[key]));
  })) return null;
  const correct = data.confusion_matrix.reduce((sum, row, index) => sum + row[index], 0),
    macro = rows.reduce((sum, row) => sum + row.f1, 0) / size;
  if (!unit(data.accuracy) || !unit(data.macro_f1) || !close(data.accuracy, correct / data.rows) ||
      !close(data.macro_f1, macro) || !close(job.metric.value, macro)) return null;
  const confusions = [];
  data.confusion_matrix.forEach((row, actual) => row.forEach((value, prediction) => {
    if (actual !== prediction && value > 0) confusions.push({
      actual: data.labels[actual], prediction: data.labels[prediction], count: value,
      fraction: value / supports[actual],
    });
  }));
  confusions.sort((a, b) => b.count - a.count || b.fraction - a.fraction || a.actual.localeCompare(b.actual));
  rows.sort((a, b) => Number(!a.support) - Number(!b.support) || a.f1 - b.f1 || a.label.localeCompare(b.label));
  return { data, rows, confusions, correct, supports };
}
function classificationDiagnosticView(job) {
  const evidence = classificationDiagnosticEvidence(job);
  if (!evidence) return "";
  const { data, rows, confusions, correct, supports } = evidence,
    percent = (value) => `${number(value * 100, 1)}%`,
    imported = job.source === "imported",
    table = rows.map((row) => `<tr data-diagnostic-class="${esc(row.label)}"><th scope="row">${esc(row.label)}<small class="diagnostic-class-predicted">Прогнозов: ${number(row.predicted)}</small>${row.support === 0 ? '<small class="diagnostic-empty-class">Нет примеров в проверке</small>' : ""}</th><td><div class="diagnostic-f1"><span class="diagnostic-bar" aria-hidden="true"><i style="width:${(row.f1 * 100).toFixed(3)}%"></i></span><strong>${number(row.f1, 3)}</strong></div></td><td>${row.support ? percent(row.recall) : "—"}</td><td>${row.predicted ? percent(row.precision) : "—"}</td><td class="diagnostic-count">${number(row.support)}</td></tr>`).join(""),
    matrixRows = data.confusion_matrix.map((row, actual) => `<tr><th scope="row">${esc(data.labels[actual])}</th>${row.map((count, prediction) => {
      const fraction = supports[actual] ? count / supports[actual] : 0,
        caption = `Истинный класс: ${data.labels[actual]}; предсказано: ${data.labels[prediction]}; ${count} из ${supports[actual]} (${percent(fraction)})`;
      return `<td class="${actual === prediction ? "diagnostic-correct" : "diagnostic-error"}" style="--cell-strength:${fraction.toFixed(4)}" title="${esc(caption)}"><span>${number(count)}</span><small>${supports[actual] ? percent(fraction) : "—"}</small></td>`;
    }).join("")}</tr>`).join("");
  return `<section class="classification-diagnostics" aria-label="Разбор ошибок классификации"><div class="diagnostic-heading"><div><h3>Где ошибается модель</h3><p>${imported ? "Отчёт участника · не проверен локально" : "На проверочных строках этого запуска"}</p></div><span class="badge ${imported ? "amber" : ""}">${number(data.labels.length)} классов</span></div><div class="diagnostic-totals"><span><b>${number(correct)}</b> верно из ${number(data.rows)}</span><span><b>${percent(data.accuracy)}</b> accuracy</span></div><div class="table-wrap diagnostic-class-table"><table aria-label="Качество по классам, от слабых к сильным"><thead><tr><th scope="col">Класс</th><th scope="col">F1</th><th scope="col" title="Recall: какую долю примеров этого класса удалось найти">Найдено</th><th scope="col" title="Precision: какая доля предсказаний этого класса верна">Точность</th><th scope="col">В проверке</th></tr></thead><tbody>${table}</tbody></table></div><div class="diagnostic-confusions"><h4>Чаще всего путает</h4>${confusions.length ? `<ul>${confusions.slice(0, 5).map((pair) => `<li><span class="diagnostic-pair"><b>${esc(pair.actual)}</b><span aria-hidden="true">→</span><b>${esc(pair.prediction)}</b></span><span class="diagnostic-pair-count"><strong>${number(pair.count)}</strong><small>${percent(pair.fraction)} класса ${esc(pair.actual)}</small></span></li>`).join("")}</ul>` : '<p class="diagnostic-perfect">Все проверочные примеры распознаны верно.</p>'}</div><details class="diagnostic-matrix"><summary>Матрица ошибок</summary><p>Строки — истинный класс, столбцы — прогноз. Цвет и проценты показывают долю внутри строки.</p><div class="table-wrap diagnostic-matrix-scroll" tabindex="0" role="region" aria-label="Матрица ошибок: строки — истинный класс, столбцы — прогноз"><table><caption class="sr-only">Количество и доля предсказаний для каждого истинного класса</caption><thead><tr><th scope="col">Истинный ↓ / прогноз →</th>${data.labels.map((label) => `<th scope="col">${esc(label)}</th>`).join("")}</tr></thead><tbody>${matrixRows}</tbody></table></div><div class="diagnostic-legend"><span>Зелёный — верно</span><span>Красный — ошибка</span></div></details></section>`;
}
function searchFeatureView(config) {
  if (!config || typeof config !== "object") return "";
  const labels = [
    config.missing_indicators === true ? "Индикаторы пропусков" : "",
    Number.isFinite(config.rare_min_count) && config.rare_min_count > 0 ? "Редкие категории" : "",
    Array.isArray(config.date_columns) && config.date_columns.length ? "Даты" : "",
    Array.isArray(config.text_columns) && config.text_columns.length ? "Текст" : "",
    config.signed_log === true ? "Логарифм" : "",
  ].filter(Boolean);
  return labels.length ? `<span class="search-feature-tags">${labels.map((label) => `<span>${label}</span>`).join("")}</span>` : "";
}
function searchStages(job, search, candidates = []) {
  const phase = search?.phase,
    verified = candidates.some((row) => Number.isFinite(row.metric)),
    finished = job.status === "completed" && job.output_valid === true,
    phaseIndex = { seed: 0, selection: 1, verification: 2, blend: 2, refit: 3, finished: 3 }[phase],
    current = finished ? 3 : Number.isInteger(phaseIndex) ? phaseIndex
      : job.stage === "refitting" ? 3 : job.stage === "validating" ? 2 : verified ? 1 : 0,
    labels = ["Быстрый результат", "Поиск", "Проверка", "Финальное обучение"];
  return `<ol class="search-stages" aria-label="Этапы автоподбора">${labels.map((label, index) => {
    const complete = finished || index < current,
      active = !finished && index === current && runningJob(job) && job.status !== "cleanup_pending";
    return `<li class="${complete ? "complete" : active ? "active" : ""}" ${active ? 'aria-current="step"' : ""}><span class="search-step-number" aria-hidden="true">${complete ? icon("check") : index + 1}</span><span>${label}<small>${complete ? "Завершён" : active ? "Сейчас" : ""}</small></span></li>`;
  }).join("")}</ol>`;
}
function adaptiveSearchView(search) {
  const controller = search?.controller;
  if (!controller || controller.strategy !== "adaptive") return '<p class="search-score-note">Фиксированный набор рецептов · прежний режим</p>';
  const reasons = {selection_budget: "Поиск завершён: оставлено время на проверку и финальное обучение", trial_limit: "Достигнут защитный лимит попыток", families_exhausted: "Доступные семейства моделей исчерпаны", search_exhausted: "Нет новых допустимых вариантов", inner_unavailable: "Недостаточно данных для внутреннего подбора", interrupted: "Поиск прерван; проверенный результат сохранён"};
  const optimizer = controller.optimizer === "optuna_tpe" ? "Optuna TPE" : controller.optimizer === "deterministic_random" ? "Случайный подбор с фиксированным seed" : "Подготовка поиска";
  return `<p class="search-score-note"><strong>Адаптивный поиск</strong> · ${esc(optimizer)} · обработано ${number(controller.observed)} из ${number(controller.proposed)} предложений · защитный лимит ${number(controller.max_trials)}</p>${controller.stop_reason ? `<p class="search-validation">${esc(reasons[controller.stop_reason] || controller.stop_reason)}</p>` : ""}`;
}
function autoSearchView(job) {
  if (job?.action !== "automl") return "";
  const search = job.search, cleanupPending = job.status === "cleanup_pending";
  if (!search || !Array.isArray(search.candidates)) return cleanupPending
    ? '<section class="search-summary"><div class="search-heading"><h3>Остановка автоподбора требует проверки</h3><span class="badge">Без API</span></div><p>Завершение процесса ещё не подтверждено. Повторите остановку.</p></section>'
    : runningJob(job)
    ? `<section class="search-summary"><div class="search-heading"><h3>Подготавливаем автоподбор</h3><span class="badge">Без API</span></div>${searchStages(job)}<p>Выбираем проверку и готовим стартовую модель.</p></section>` : "";
  // Match the bounded plan accepted by app/search_summary.py, including metric recipes.
  const candidates = search.candidates.slice(0, 528),
    planned = Number.isInteger(search.planned) && search.planned > 0 ? Math.min(search.planned, 528) : candidates.length,
    done = candidates.filter((row) => row.status === "completed").length,
    verified = candidates.filter((row) => Number.isFinite(row.metric)).length,
    selectionCount = candidates.filter((row) => Number.isFinite(row.selection_metric)).length,
    expanded = candidates.some((row) => Object.hasOwn(row, "selection_metric")),
    finished = job.status === "completed" && job.output_valid === true,
    best = candidates.find((row) => row.id === search.best_candidate && row.status === "completed" && row.selected === true && Number.isFinite(row.metric)),
    status = cleanupPending ? "Остановка автоподбора требует проверки"
      : search.status === "fallback" && finished && best
      ? "Поиск прерван; сохранён проверенный результат"
      : search.status === "time_limit" && finished && best
      ? "Лимит времени · лучший сохранён"
      : search.status === "failed" ? "Автоподбор завершился с ошибкой"
      : finished && best ? "Лучший вариант сохранён" : runningJob(job) ? "Подбираем модель" : "Автоподбор остановлен",
    metricName = job.metric?.name || currentProfile()?.metric?.name || "Метрика",
    labels = { completed: "Оценён", failed: "Ошибка", timed_out: "Лимит времени", skipped: "Пропущен", running: "Выполняется", pending: "В очереди" },
    phases = { seed: "Стартовая модель", selection: "Поиск", verification: "Проверка", refit: "Финальное обучение", blend: "Ансамбль" },
    reasons = {
      timeout: "Лимит времени", memory_limit: "Лимит памяти", execution_failed: "Ошибка обучения",
      invalid_evidence: "Проверка результата не пройдена", not_selected: "Не прошёл отбор",
      not_improved: "Без улучшения", insufficient_budget: "Не хватило времени",
      refit_failed: "Ошибка финального обучения", optional_dependency_unavailable: "Библиотека не установлена",
    };
  return `<section class="search-summary" aria-label="Ход автоподбора"><div class="search-heading"><div><h3>${esc(status)}</h3><p>Оценено ${number(done)}${search.controller ? "" : ` / ${number(planned)}`} вариантов${expanded ? ` · подбор ${number(selectionCount)} · проверка ${number(verified)}` : ""}</p></div>${badge("Без API", "green")}</div>${searchStages(job, search, candidates)}${adaptiveSearchView(search)}<progress class="search-progress" max="${Math.max(1, planned)}" ${search.controller && runningJob(job) ? 'aria-label="Адаптивный поиск продолжается"' : `value="${Math.min(done, planned)}" aria-label="Оценённые варианты"`}></progress>${search.validation_reason ? `<p class="search-validation">${esc(search.validation_reason)}</p>` : ""}<p class="search-score-note">${esc(metricName)} · подбор внутри обучающей части, проверка на отложенных строках.</p>${searchFeatureView(job.feature_config)}<div class="table-wrap" tabindex="0" role="region" aria-label="Таблица попыток автоподбора"><table class="search-candidates" aria-label="Кандидаты автоподбора"><thead><tr><th scope="col">Вариант</th><th scope="col">Подбор</th><th scope="col">Проверка</th><th scope="col">Итог</th><th scope="col">Время</th><th scope="col">Пик RAM</th></tr></thead><tbody>${candidates.map((row) => {
    const selected = best?.id === row.id,
      verifiedValue = Number.isFinite(row.verification_metric) ? row.verification_metric : row.metric,
      statusLabel = row.status === "skipped" && search.status === "running" && runningJob(job)
        ? "В очереди" : row.status === "completed" && !Number.isFinite(row.metric) ? "Подбор завершён" : labels[row.status] || "Статус не указан",
      memory = row.resources?.peak_rss_bytes,
      memoryText = Number.isFinite(memory) && memory >= 0 ? `${number(memory / (1024 * 1024), 1)} МБ` : "—",
      reason = reasons[row.rejection_reason] || "";
    return `<tr class="${selected ? "search-selected" : ""}"><th scope="row"><span>${esc(row.label || row.model || row.id)}</span><small>${esc(row.model || "")}${phases[row.phase] ? ` · ${phases[row.phase]}` : ""}</small>${searchFeatureView(row.feature_config)}</th><td class="metric-cell">${number(row.selection_metric, 6)}</td><td class="metric-cell">${number(verifiedValue, 6)}</td><td>${selected ? badge(finished ? "Выбран" : "Пока лучший", "green") : badge(statusLabel, row.status === "failed" || row.status === "timed_out" ? "amber" : "")}${reason ? `<small>${esc(reason)}</small>` : ""}${row.error ? `<details class="search-error"><summary>Подробности</summary><p>${esc(row.error)}</p></details>` : ""}</td><td class="search-duration">${duration(row.elapsed_seconds)}</td><td class="search-memory">${memoryText}${row.resources?.available === false ? "<small>Не измерялось</small>" : ""}</td></tr>`;
  }).join("") || '<tr><td colspan="6">Готовим первый вариант…</td></tr>'}</tbody></table></div></section>`;
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
      `<div class="run-detail-header"><div><h2>${esc(jobTitle(job))}</h2><p>${esc(date(job.started_at))} · ${esc(job.author || "Без автора")}</p></div>${statusBadge(job.agent_status === "fallback" ? "fallback" : job.status)}</div>${job.hypothesis ? `<p class="job-hypothesis">${esc(job.hypothesis)}</p>` : ""}<div class="job-stage">${runningJob(job) ? '<span class="spinner"></span>' : icon(job.status === "completed" ? "check" : stopped ? "clock" : "missing")}<span>${esc(job.status === "cancelled" ? "Остановлен пользователем" : job.status === "failed" ? "Запуск завершился с ошибкой" : STAGE_NAMES[job.stage] || "Подготовка")}</span><span style="margin-left:auto">${duration(job.elapsed_seconds)}</span></div>${job.error ? `<div class="warning-box">${esc(job.error)}</div>` : ""}${job.action === "agent" ? researchView(job.research, { status: job.agent_status, imported: job.source === "imported" }) : ""}${autoSearchView(job)}${validMetric ? jobMetricView(job) : job.status === "completed" && job.action === "agent" ? `<div class="info-box">${job.agent_status === "fallback" ? "Сохранён резервный CSV. Исследование не завершилось штатно." : "Процесс Qwen завершён."} Результаты проверки качества и выбора модели смотрите в журнале агента. В сравнение запусков эта метрика пока не включена.</div>` : ""}<div class="job-actions">${currentLocalResult(job) && isFixedRecipe(job.action) ? `<button class="button" data-edit-job="${esc(job.id)}">Изменить настройки</button>` : ""}${runningJob(job) ? `<button class="button danger" data-action="stop-job">${job.status === "cleanup_pending" ? "Повторить остановку" : "Остановить"}</button>` : ""}${job.status === "completed" && job.output_valid === true ? `<a class="button" href="/api/jobs/${encodeURIComponent(job.id)}/submission${task ? "?task_id=" + encodeURIComponent(task) : ""}" download>${icon("download")} Скачать CSV</a>${job.published ? badge("CSV используется", "green") : `<button class="button primary" data-action="publish-job" data-publish-id="${esc(job.id)}">Использовать CSV</button>`}` : ""}${lightgbmStudySource(job, task) && currentLocalResult(job) ? `<button class="button" data-lightgbm-study-source="${esc(job.id)}" data-task-id="${esc(task)}">Сравнить обучение LightGBM</button>` : ""}${jobBundleLink(job)}${job.action === "prepare" && job.status === "completed" ? '<button class="button primary" data-action="open-data">Открыть данные</button>' : ""}${job.action === "agent" ? '<button class="button" data-action="inspect-agent">Код агента</button>' : ""}<span class="subtle-text">${job.action === "agent" ? "Qwen · общий API-бюджет" : "Без API"}</span></div>${trainedModelView(job)}${trainingDeviceEvidenceView(job)}${job.api ? `<div class="usage-row">${usageView(job.api)}</div>` : ""}<section class="panel job-logs"><div class="panel-head"><h2>Журнал запуска</h2><span class="job-log-time" id="job-refresh-time">Обновляется автоматически</span></div><div class="code-tabs" role="tablist" aria-label="Вывод процесса"><button class="code-tab ${state.jobLogKind === "stdout" ? "active" : ""}" data-job-log="stdout" role="tab" aria-selected="${state.jobLogKind === "stdout"}">Вывод</button><button class="code-tab ${state.jobLogKind === "stderr" ? "active" : ""}" data-job-log="stderr" role="tab" aria-selected="${state.jobLogKind === "stderr"}">Ошибки</button></div><pre id="job-log-output" class="code-content" role="tabpanel"></pre><div class="code-footnote">${detail.truncated ? "Показан ограниченный фрагмент журнала" : "Вывод процесса"} · ${esc(job.id?.slice(0, 8) || "")}</div></section>`;
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
function launchValidationConfig(action, profile, editReference = null) {
  const numeric = (profile?.columns || []).filter(
    (column) => column.role === "feature" && column.dtype === "numeric",
  );
  const fallback = {
    strategy: "random",
    time_column: numeric.find((column) => column.name === "Year_Factor")?.name || "",
  };
  if (action === "automl") return { ...fallback, strategy: "auto" };
  if (action === "baseline" && !editReference) return fallback;
  const eligible = (job) =>
    currentLocalResult(job) &&
    isOfflineRecipe(job.action);
  const selected =
    state.jobDetail?.job?.id === state.selectedJob
      ? state.jobDetail.job
      : state.jobs.find((job) => job.id === state.selectedJob);
  const reference = eligible(editReference) ? editReference : eligible(selected)
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
function validationFields(profile, selected, action = "baseline") {
  const numeric = (profile?.columns || []).filter(
    (column) => column.role === "feature" && column.dtype === "numeric",
  );
  const options = [["random", "Случайные 80/20", "С фиксированным seed 42", true]];
  if (profile?.task_type === "regression")
    options.push(["time", "Последний период", "Ранние → последний", numeric.length > 0]);
  if (profile?.task_type === "classification")
    options.push(["grouped", "Разделить дубли", "Копии только в одной части", true]);
  return `<fieldset id="validation-controls" class="validation-controls"><legend>Проверка качества</legend><div class="validation-options"><label class="validation-option" id="validation-auto-option" ${action === "automl" ? "" : "hidden"}><input type="radio" name="validation_strategy" value="auto" ${selected.strategy === "auto" ? "checked" : ""} ${action === "automl" ? "" : "disabled"}><span><strong>Автоматически</strong><small>С учётом периодов и дублей</small></span></label>${options.map(([value, label, description, enabled]) => `<label class="validation-option"><input type="radio" name="validation_strategy" value="${value}" ${selected.strategy === value ? "checked" : ""} ${enabled ? "" : "disabled"}><span><strong>${label}</strong><small>${enabled ? description : "Нет числовых признаков"}</small></span></label>`).join("")}</div><label class="form-field validation-time-column" id="validation-time-field" hidden>Столбец периода<select name="time_column" required disabled><option value="">Выберите столбец периода</option>${numeric.map((column) => `<option value="${esc(column.name)}" ${selected.time_column === column.name ? "selected" : ""}>${esc(column.name)}</option>`).join("")}</select></label></fieldset>`;
}
function trainingDeviceControlsHTML() {
  const gpu = state.trainingDevices?.gpu, ready = gpu?.available === true;
  const description = state.trainingDevicesBusy ? "Проверяем доступность GPU через CatBoost…"
    : !gpu ? "Проверка видеокарты ещё не выполнена. CPU доступен сразу."
    : ready ? `${gpu.device_name || gpu.label || "GPU"} · CatBoost видит устройство. Проверка доступности не запускает обучение.`
    : gpu.reason || "CatBoost пока не видит доступную видеокарту.";
  return `<div class="training-device-picker"><label class="form-field">Устройство обучения<select name="training_device" id="training-device"><option value="cpu" ${state.launchTrainingDevice !== "gpu" ? "selected" : ""}>CPU · 8 потоков</option><option value="gpu" ${state.launchTrainingDevice === "gpu" ? "selected" : ""} ${ready ? "" : "disabled"}>GPU · CatBoost${ready ? "" : " · недоступно"}</option></select></label><button class="button small" type="button" data-action="training-device-probe" ${state.trainingDevicesBusy ? "disabled" : ""}>${state.trainingDevicesBusy ? '<span class="spinner"></span> Проверяем…' : "Проверить видеокарту"}</button><p class="campaign-note" role="status">${esc(description)}</p><p class="campaign-note">GPU — отдельный локальный эксперимент. Автоматического перехода на CPU при ошибке нет. Сравнение LightGBM и конкурсная проверка используют CPU.</p></div>`;
}
function renderTrainingDeviceControls() {
  const container = $("#training-device-controls");
  if (!container) return;
  container.innerHTML = trainingDeviceControlsHTML();
  container.hidden = state.launchModelRecipe !== "catboost";
}
async function refreshTrainingDevices() {
  if (state.trainingDevicesBusy) return;
  state.trainingDevicesBusy = true; renderTrainingDeviceControls();
  try {
    const devices = await api("/api/training-devices");
    state.trainingDevices = devices && typeof devices.gpu?.available === "boolean" ? devices
      : {gpu: {available: false, reason: "Сервер не подтвердил доступность GPU."}};
  } catch (error) { state.trainingDevices = {gpu: {available: false, reason: error.message}}; }
  finally { state.trainingDevicesBusy = false; renderTrainingDeviceControls(); }
}
function updateLaunchFields() {
  const form = $("#launch-form"),
    value = $('input[name="recipe"]:checked', form)?.value,
    offline = isOfflineRecipe(value),
    fixedRecipe = isFixedRecipe(value),
    timed = value === "agent" || offline,
    autoOption = $('input[name="validation_strategy"][value="auto"]', form);
  if (value !== "automl" && autoOption.checked) {
    autoOption.checked = false;
    $('input[name="validation_strategy"][value="random"]', form).checked = true;
  }
  autoOption.disabled = value !== "automl";
  $("#validation-auto-option").hidden = value !== "automl";
  const strategy = $('input[name="validation_strategy"]:checked', form)?.value;
  if (state.launchModelRecipe !== value) {
    state.launchTrainingDevice = "cpu";
    if (state.launchModelRecipe)
      $("#launch-title").textContent = {
        automl: "Подобрать лучшую модель", baseline: "Измерить стартовое качество",
        extra_trees: "Сравнить другую модель", catboost: "Обучить CatBoost",
        lightgbm: "Обучить LightGBM", linear: "Обучить линейную модель", agent: "Исследование с Qwen",
      }[value] || "Новый запуск";
    saveModelDraft();
    mountModelSettings(value);
  }
  $("#launch-cost").textContent =
    value === "agent"
      ? "Qwen · расходует общий API-бюджет"
      : "Без API · обучение на этом компьютере";
  $("#launch-cost").classList.toggle("paid", value === "agent");
  $("#qwen-time-field").hidden = !timed;
  $("#time-budget-label").textContent = value === "automl" ? "Лимит автоподбора, минут" : value === "agent" ? "Время исследования Qwen, минут" : "Лимит обучения, минут";
  $("#automl-plan").hidden = value !== "automl";
  $("#search-strategy-controls").hidden = value !== "automl";
  $("#hypothesis-label").textContent =
    value === "agent" ? "Гипотеза для Qwen" : "Заметка к эксперименту";
  form.elements.minutes.disabled = !timed;
  $("#validation-controls").hidden = !offline;
  $("#validation-controls").disabled = !offline;
  $("#validation-time-field").hidden = !offline || strategy !== "time";
  form.elements.time_column.disabled = !offline || strategy !== "time";
  $("#model-settings").hidden = !fixedRecipe;
  if ($("#model-fields")) $("#model-fields").disabled = !fixedRecipe;
  renderTrainingDeviceControls();
}
async function showLaunch(action = "automl", editReference = null) {
  if (nativeRunning(state.nativeActiveQueue) || campaignRunning(state.campaignActive)) {
    toast("Очередь CPU занята серией. Продолжите её или остановите в соответствующем разделе.");
    return;
  }
  if (action === "prepare") {
    startPrepare();
    return;
  }
  if (!state.task) return;
  if (editReference && (!currentLocalResult(editReference) || editReference.action !== action)) {
    toast("Выберите локальный завершённый запуск на текущих данных.");
    return;
  }
  const task = state.task,
    generation = ++state.launchGeneration,
    viewGeneration = state.generation;
  state.launchTask = task;
  $("#launch-title").textContent =
    action === "automl"
      ? "Подобрать лучшую модель"
      : action === "agent"
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
    if (!profile || !profile.model_options || (fingerprint && profile.dataset_fingerprint !== fingerprint))
      profile = await api(`${apiBase(task)}/profile`);
  } catch (error) {
    if (state.task === task && generation === state.launchGeneration && viewGeneration === state.generation)
      toast("Не удалось подготовить запуск: " + error.message);
    return;
  }
  if (state.task !== task || generation !== state.launchGeneration || viewGeneration !== state.generation) return;
  state.profiles.set(task, profile);
  state.launchProfile = profile;
  state.launchModelRecipe = null;
  state.launchTrainingDevice = "cpu";
  state.launchFeatureConfigs = editReference?.feature_config ? {[action]: structuredClone(editReference.feature_config)} : {};
  state.launchOriginalConfigs = editReference?.model_config ? {[action]: structuredClone(editReference.model_config)} : {};
  state.launchModelDrafts = Object.fromEntries(FIXED_RECIPES.flatMap((recipe) => {
    const schema = modelSchema(profile, recipe);
    return schema ? [[recipe, modelDraft(schema, editReference?.action === recipe ? editReference.model_config : null)]] : [];
  }));
  if (isFixedRecipe(action) && !modelSchema(profile, action)) action = "automl";
  const selectedValidation = launchValidationConfig(action, profile, editReference);
  let author = "";
  try {
    author = localStorage.getItem("alfa-author") || "";
  } catch {}
  if (!author) author = editReference?.author || "";
  if (editReference) $("#launch-title").textContent = `Новая попытка · ${ACTION_NAMES[action]}`;
  const recipes = launchRecipeOptions(profile);
  $("#launch-form").innerHTML = `<fieldset class="recipe-options"><legend class="sr-only">Способ обучения</legend>${recipes
    .map(
      ([value, label, desc]) =>
        `<label class="recipe-option"><input type="radio" name="recipe" value="${esc(value)}" ${value === action ? "checked" : ""} ${value === "agent" && !ready ? "disabled" : ""}><strong>${esc(label)}</strong><small>${esc(desc)}</small></label>`,
    )
    .join(
      "",
    )}</fieldset><div id="automl-plan" class="automl-plan" ${action === "automl" ? "" : "hidden"}><span>${icon("activity")} Быстрый результат</span><span>${icon("search")} Поиск</span><span>${icon("test")} Проверка</span><span>${icon("check")} Финальное обучение</span></div><section id="search-strategy-controls" ${action === "automl" ? "" : "hidden"}><label class="form-field">Как искать модель<select name="search_strategy"><option value="adaptive" selected>Адаптивный поиск · рекомендуется</option><option value="legacy">Фиксированный набор · прежний режим</option></select></label><p class="launch-note">Адаптивный поиск учится на результатах попыток и увеличивает обучение перспективных моделей. До 256 новых вариантов в пределах времени, с резервом на проверку и финальное обучение. Перезапуск приложения не продолжает оборванный поиск.</p></section>${validationFields(profile, selectedValidation, action)}<details class="model-settings" id="model-settings"></details><section id="training-device-controls" hidden></section>${editReference && !editReference.model_config ? '<p class="model-schema-note">Параметры прежнего запуска не сохранены. Загружены стандартные.</p>' : ""}<div class="form-grid"><label class="form-field">Ваше имя<input name="author" maxlength="80" value="${esc(author)}" placeholder="Необязательно" autocomplete="name"></label><label class="form-field" id="qwen-time-field"><span id="time-budget-label">${action === "automl" ? "Лимит автоподбора, минут" : action === "agent" ? "Время исследования Qwen, минут" : "Лимит обучения, минут"}</span><input name="minutes" type="number" min="1" max="175" value="${action === "automl" ? 10 : action === "agent" ? 20 : 30}" required></label><label class="form-field full"><span id="hypothesis-label">${action === "agent" ? "Гипотеза для Qwen" : "Заметка к эксперименту"}</span><textarea name="hypothesis" maxlength="500" placeholder="Необязательно">${esc(editReference?.hypothesis || "")}</textarea></label></div><p class="launch-note">Каждый запуск сохранит свой результат. Текущий CSV изменится только после выбора «Использовать CSV».</p><p id="launch-error" class="form-error" role="alert"></p><div class="form-actions"><span id="launch-cost" class="form-cost">${action === "agent" ? "Qwen · расходует общий API-бюджет" : "Без API · обучение на этом компьютере"}</span><button class="button primary" type="submit" id="launch-submit">Запустить ${icon("activity")}</button></div>`;
  $("#launch-form").onchange = (event) => {
    if (event.target.name === "training_device") state.launchTrainingDevice = event.target.value;
    updateLaunchFields();
  };
  $("#launch-form").oninput = () => { saveModelDraft(); updateModelSummary(); };
  updateLaunchFields();
  if (editReference) $("#model-settings").open = true;
  $("#launch-dialog").showModal();
  if (!state.trainingDevices) refreshTrainingDevices();
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
$("#launch-form").addEventListener("invalid", (event) => {
  if (event.target.closest("#model-settings")) $("#model-settings").open = true;
}, true);
$("#launch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget,
    button = $("#launch-submit");
  if (button.disabled) return;
  if (nativeRunning(state.nativeActiveQueue) || campaignRunning(state.campaignActive)) {
    $("#launch-error").textContent = "Серия занимает очередь CPU. Дождитесь завершения или остановите её.";
    return;
  }
  const action = $('input[name="recipe"]:checked', form)?.value,
    author = form.elements.author.value.trim(),
    hypothesis = form.elements.hypothesis.value.trim(),
    minutes = Number(form.elements.minutes.value),
    task = state.launchTask;
  if (!action || !Number.isInteger(minutes) || minutes < 1 || minutes > 175) {
    $("#launch-error").textContent = "Укажите целое число минут от 1 до 175.";
    return;
  }
  let validation, model_params;
  if (isOfflineRecipe(action)) {
    const strategy = $('input[name="validation_strategy"]:checked', form)?.value;
    if (!["random", "time", "grouped", ...(action === "automl" ? ["auto"] : [])].includes(strategy)) {
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
    const schema = action === "automl" ? null : modelSchema(state.launchProfile, action);
    if (schema) {
      try { model_params = parseModelParams(schema, readModelDraft()); }
      catch (error) {
        $("#model-settings").open = true;
        $("#launch-error").textContent = error.message;
        return;
      }
    }
  }
  const training_device = action === "catboost" ? state.launchTrainingDevice : "cpu";
  if (!["cpu", "gpu"].includes(training_device) || training_device === "gpu" && state.trainingDevices?.gpu?.available !== true) {
    $("#launch-error").textContent = "Доступность GPU не подтверждена. Повторите проверку видеокарты или выберите CPU.";
    return;
  }
  button.disabled = true;
  $("#launch-error").textContent = "";
  try {
    const response = await post("/api/jobs", launchPayload({
      task_id: task,
      action,
      author,
      hypothesis,
      minutes,
      validation,
      model_params,
      model_config: state.launchOriginalConfigs[action] && model_params ? {...structuredClone(state.launchOriginalConfigs[action]), parameters: model_params} : null,
      feature_config: state.launchFeatureConfigs[action],
      training_device,
      search_strategy: form.elements.search_strategy?.value || "adaptive",
    }));
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
  } finally {
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
  const groups = comparisonGroups(reports), winners = new Set();
  let groupCount = 0;
  for (const {rows,best} of groups) {
    if (rows.length < 2) continue;
    groupCount++;
    rows.filter((row) => row.metric.value === best.metric.value).forEach((row) => winners.add(row.id));
  }
  return {winners,groupCount};
}
function experimentsPanel(reports) {
  const { winners, groupCount } = comparisonWinners(reports);
  return `<section class="panel" id="experiment-history"><div class="panel-head result-history-head"><div><h2>Эксперименты команды</h2><p>Метрики, гипотезы и авторы</p></div><div class="action-buttons"><button class="button small" data-launch="automl">Автоподбор</button></div></div>${reports.length ? `<div class="table-wrap experiment-table"><table aria-label="Результаты экспериментов"><thead><tr><th scope="col">Модель и гипотеза</th><th scope="col">Метрика</th><th scope="col">Проверка</th><th scope="col">Автор</th><th scope="col">Время / статус</th><th scope="col">Результат</th></tr></thead><tbody>${reports.map((r) => `<tr class="${r.source === "imported" ? "experiment-row-imported" : winners.has(r.id) ? "experiment-row-best" : ""}"><td><span class="model-name">${esc(jobTitle(r))}<small>${esc(r.model || "")}</small></span>${modelParameterView(r, true)}<span class="report-hypothesis">${esc(r.hypothesis || "Гипотеза не указана")}</span><span class="report-source ${r.source === "imported" ? "source-imported" : ""}">${r.source === "imported" ? "От участника · не проверено локально" : "На этом компьютере"}</span>${r.action === "agent" ? researchProvenance(r.research, r.source === "imported") : ""}${r.dataset_fingerprint && state.currentDatasetFingerprint && r.dataset_fingerprint !== state.currentDatasetFingerprint ? badge("Другая версия данных", "amber") : ""}</td><td class="metric-cell">${r.metric && Number.isFinite(r.metric.value) ? number(r.metric.value, 6) : "—"}<small>${esc(r.metric?.name || "Нет метрики")}</small>${winners.has(r.id) ? badge("Лучший в группе", "green") : ""}</td><td class="validation-cell">${validationView(r, true)}</td><td>${esc(r.author || "Не указан")}</td><td>${duration(r.elapsed_seconds)}<br>${statusBadge(r.agent_status === "fallback" ? "fallback" : r.status)}</td><td>${r.source === "local" && r.status === "completed" && r.output_valid === true && r.dataset_fingerprint === state.currentDatasetFingerprint ? `<div class="result-row-actions">${r.published ? badge("CSV используется", "green") : `<button class="button small" data-action="publish-job" data-publish-id="${esc(r.id)}">Использовать CSV</button>`}${jobBundleLink(r, "CSV и отчёт ZIP", true)}${isFixedRecipe(r.action) ? `<button class="button small" data-edit-job="${esc(r.id)}">Изменить настройки</button>` : ""}</div>` : r.source === "imported" ? '<span class="subtle-text">Только отчёт</span>' : '<span class="subtle-text">—</span>'}</td></tr>`).join("")}</tbody></table></div><div class="comparison-note">${groupCount ? `${groupCount} групп сравнения: одна задача, схема, метрика и одинаковые проверочные строки.` : "Пока нет пары локальных результатов с одинаковой схемой проверки."} Метрики разных схем не ранжируются вместе.</div>` : '<div class="job-empty"><p>Запустите автоподбор — появятся первая метрика и лучший вариант.</p><button class="button primary" data-launch="automl">Автоподбор</button></div>'}</section>`;
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
  const [result, experiments, profile] = await Promise.allSettled([
    api(`${apiBase(task)}/result`),
    api(`${apiBase(task)}/experiments`),
    modelSchema(currentProfile(), "baseline")
      ? Promise.resolve(currentProfile()) : api(`${apiBase(task)}/profile`),
  ]);
  if (generation !== state.generation || task !== state.task) return;
  if (profile.status === "fulfilled") state.profiles.set(task,profile.value);
  state.resultData = result.status === "fulfilled" ? result.value : {submission:{},baseline_error:result.reason.message};
  const response = experiments.status === "fulfilled" ? experiments.value : {experiments:[],dataset_fingerprint:state.workflow?.dataset_fingerprint};
  state.currentDatasetFingerprint = response.dataset_fingerprint || null;
  state.experimentReports = response.experiments || [];
  state.experimentFingerprint = resultFingerprint(state.resultData,response);
  main.innerHTML = heading("ИТОГИ И ПРОВЕРКА","Результат исследования","Сравните эксперименты и выберите CSV для платформы.") +
    taskStrip(taskInfo()) + resultGroupsPanel(state.experimentReports) +
    submissionPanel(state.resultData,state.experimentReports) + downloadsPanel() +
    (experiments.status === "rejected" ? `<div class="warning-box">${esc(experiments.reason.message)}</div>` : "") +
    experimentsPanel(state.experimentReports) + legacyBaselinePanel(state.resultData);
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
  if (file.size > (zipped ? 20 : 8) * 1024 * 1024) {
    toast(
      zipped
        ? "Пакет слишком большой. Максимум 20 МБ."
        : "Отчёт слишком большой. Максимум 8 МБ.",
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
  if (["overview", "campaigns", "native"].includes(state.view)) return;
  if (document.hidden || state.jobPollBusy || state.pollBusy) return;
  await syncNativeActivity();
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
        const [response, result] = await Promise.all([
          api(`${apiBase(task)}/experiments`),
          api(`${apiBase(task)}/result`),
        ]);
        if (generation !== state.generation || task !== state.task) return;
        const fingerprint = resultFingerprint(result, response);
        if (
          fingerprint !== state.experimentFingerprint &&
          $("#experiment-history")
        ) {
          state.experimentFingerprint = fingerprint;
          state.currentDatasetFingerprint =
            response.dataset_fingerprint || null;
          state.experimentReports = response.experiments || [];
          state.resultData = result;
          if (!modelSchema(currentProfile(), "baseline")) {
            const profile = await api(`${apiBase(task)}/profile`).catch(() => null);
            if (generation !== state.generation || task !== state.task) return;
            if (profile) state.profiles.set(task, profile);
          }
          $("#result-groups").outerHTML = resultGroupsPanel(state.experimentReports);
          $("#current-submission").outerHTML = submissionPanel(result, state.experimentReports);
          const legacyOpen = $("#legacy-result")?.open;
          $("#legacy-result").outerHTML = legacyBaselinePanel(result);
          $("#legacy-result").open = legacyOpen;
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
            `<option value="${esc(item.id)}">${esc(taskName(item))}</option>`,
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
