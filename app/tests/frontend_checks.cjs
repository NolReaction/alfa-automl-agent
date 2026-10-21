"use strict";

// Only the browser shell is stubbed. The checked functions come from app.js.
// No packages, network, browser process, or filesystem writes are needed.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const noop = () => {};
const element = {
  innerHTML: "", textContent: "", value: "", disabled: false, hidden: false,
  dataset: {}, elements: {}, options: [], children: [],
  classList: { add: noop, remove: noop, toggle: noop },
  addEventListener: noop, setAttribute: noop, removeAttribute: noop,
  append: noop, insertAdjacentHTML: noop, close: noop, showModal: noop,
  querySelector: () => element, querySelectorAll: () => [],
};
const context = vm.createContext({
  console,
  document: {
    hidden: true, activeElement: null,
    querySelector: () => element, querySelectorAll: () => [],
    addEventListener: noop, createElement: () => element, body: element,
  },
  window: { addEventListener: noop },
  location: { hash: "" }, history: { replaceState: noop }, navigator: {},
  localStorage: { getItem: () => null, setItem: noop },
  setInterval: () => 0, setTimeout: () => 0, clearTimeout: noop,
  URLSearchParams, structuredClone,
  fetch: async (url) => ({
    ok: true,
    json: async () => url === "/api/tasks" ? { tasks: [] } : { api: { ready: true } },
  }),
});
vm.runInContext(fs.readFileSync(process.argv[3], "utf8"), context, {
  filename: process.argv[3], timeout: 5000,
});
const get = (name) => vm.runInContext(name, context);
const plain = (value) => JSON.parse(JSON.stringify(value));

const schema = {
  model: "RandomForestClassifier",
  defaults: { n_estimators: 120, max_depth: 18, learning_rate: 0.1, class_weight: "balanced", max_features: "sqrt" },
  fields: [
    { name: "n_estimators", label: "Деревьев", type: "integer", min: 10, max: 500, step: 1 },
    { name: "max_depth", label: "Глубина", type: "integer", min: 1, max: 64, nullable: true },
    { name: "learning_rate", label: "Скорость", type: "number", min: 0.01, max: 1, step: 0.01 },
    { name: "class_weight", label: "Вес классов", type: "select", options: [{ value: null, label: "Без весов" }, { value: "balanced", label: "Баланс" }] },
    { name: "max_features", label: "Признаков", type: "select", options: [{ value: "sqrt", label: "Корень" }, { value: 0.5, label: "Половина" }, { value: 1.0, label: "Все" }] },
  ],
};
const profile = {
  id: "case", task_type: "classification", dataset_fingerprint: "current",
  columns: [{ name: "x", role: "feature", dtype: "numeric" }],
  model_options: { baseline: schema },
};
function setup(currentProfile = profile) {
  const state = get("state");
  state.task = currentProfile.id;
  state.tasks = [currentProfile];
  state.profile = currentProfile;
  state.profiles.clear();
  state.profiles.set(currentProfile.id, currentProfile);
  state.currentDatasetFingerprint = currentProfile.dataset_fingerprint;
  state.workflow = { dataset_fingerprint: currentProfile.dataset_fingerprint };
  state.jobs = []; state.experimentReports = []; state.jobDetail = null;
  state.selectedJob = null;
  return state;
}
const job = (overrides = {}) => ({
  id: "job", task_id: "case", source: "local", action: "baseline", status: "completed",
  dataset_fingerprint: "current", comparison_key: "grouped-check", output_valid: true,
  started_at: "2026-10-20T12:00:00Z", model: "RandomForestClassifier", published: false,
  metric: { name: "Macro F1", value: 0.5936, higher_is_better: true },
  validation: { strategy: "grouped" },
  split: { strategy: "grouped", validation_fraction: 0.2, train_groups: 80, validation_groups: 20, group_overlap: 0 },
  ...overrides,
});
const rawValues = () => ({
  n_estimators: "160", max_depth: "", learning_rate: "0.2",
  class_weight: "null", max_features: "0.5",
});
function diagnosticJob(overrides = {}) {
  const normal = 180 / 193, other = 10 / 13, macro = (normal + other) / 3;
  return job({
    metric: { name: "Macro F1", value: macro, higher_is_better: true },
    split: { strategy: "grouped", validation_rows: 110 },
    classification_diagnostics: {
      version: 1, kind: "classification", labels: ["Normal", "Worms", "Other"], rows: 110,
      confusion_matrix: [[90, 10, 0], [2, 0, 2], [1, 0, 5]],
      per_class: [
        { precision: 90 / 93, recall: 0.9, f1: normal, support: 100, predicted: 93 },
        { precision: 0, recall: 0, f1: 0, support: 4, predicted: 10 },
        { precision: 5 / 7, recall: 5 / 6, f1: other, support: 6, predicted: 7 },
      ],
      accuracy: 95 / 110, macro_f1: macro,
    },
    ...overrides,
  });
}

const cases = {
  groups() {
    const state = setup();
    state.resultData = { baseline: { metric: { value: 0.999999 } } };
    const random = job({ id: "random", comparison_key: "random-check", split: { strategy: "random" }, metric: { name: "Macro F1", value: 0.61964, higher_is_better: true } });
    const grouped = job({ id: "grouped" });
    const lower = job({ id: "lower", metric: { ...grouped.metric, value: 0.5889 } });
    const groups = get("comparisonGroups")([random, grouped, lower]);
    assert.equal(groups.length, 2, "A random score must not absorb a grouped evaluation");
    assert.equal(groups.find(g => g.rows.some(r => r.id === "grouped")).best.id, "grouped");
    assert.equal(groups.find(g => g.rows.some(r => r.id === "random")).best.id, "random");
    const latest = job({ id: "latest", started_at: "2026-10-20T13:00:00Z" });
    assert.equal(get("comparisonGroups")([grouped, latest])[0].best.id, "latest", "A tied score should resolve to the most recent run");
    const html = get("resultGroupsPanel")([random, grouped, lower]);
    assert.match(html, /Случайн/);
    assert.match(html, /групп/);
    assert.doesNotMatch(html, /0[,.]999999/, "The standalone terminal baseline is not a current experiment");
  },
  provenance() {
    setup();
    const valid = job();
    const polluted = [valid,
      job({ id: "import", source: "imported", metric: { ...valid.metric, value: 1 } }),
      job({ id: "stale", dataset_fingerprint: "old" }),
      job({ id: "other-task", task_id: "other" }),
      job({ id: "running", status: "running" }),
      job({ id: "missing-key", comparison_key: null }),
      job({ id: "bad-number", metric: { ...valid.metric, value: NaN } }),
    ];
    const groups = get("comparisonGroups")(polluted);
    assert.equal(groups.length, 1);
    assert.deepEqual(plain(groups[0].rows.map(r => r.id)), [valid.id]);
    const rmse = job({ id: "rmse", metric: { name: "RMSE", value: 44, higher_is_better: false } });
    const better = job({ ...rmse, id: "better", metric: { ...rmse.metric, value: 40 } });
    assert.equal(get("comparisonGroups")([rmse, better])[0].best.id, "better", "Regression selects smaller RMSE");
    assert.equal(get("comparisonGroups")([valid, rmse]).length, 2, "Metrics with distinct meanings stay separate");
  },
  selected_csv() {
    const state = setup();
    const best = job({ id: "best", metric: { name: "Macro F1", value: 0.7, higher_is_better: true } });
    const selected = job({ id: "selected", published: true });
    state.workflow.published_job_id = selected.id;
    assert.equal(get("selectedSubmissionJob")([best, selected]).id, selected.id);
    assert.equal(get("comparisonGroups")([best, selected])[0].best.id, best.id);
    const panel = get("submissionPanel")({ submission: { exists: true, valid: true, path: "submission.csv", size_bytes: 1024 } }, [best, selected]);
    assert.match(panel, /data-open-job="selected"/);
    assert.match(panel, /0,5936/);
    assert.doesNotMatch(panel, /0,7(?:\D|$)/, "The current CSV must not borrow a higher score from another run");
    assert.equal(get("selectedSubmissionJob")([best]), null, "Unknown CSV provenance must not be inferred from quality");
    state.workflow.published_job_id = null;
    assert.equal(get("selectedSubmissionJob")([job({ source: "imported", published: true })]), null);
    assert.equal(get("selectedSubmissionJob")([job({ dataset_fingerprint: "old", published: true })]), null);
  },
  terminal_report() {
    setup();
    const html = get("legacyBaselinePanel")({ baseline: {
      metric: { name: "Macro F1", value: 0.61964, higher_is_better: true },
      model: { name: "RandomForestClassifier" }, split: { seed: 42, validation_fraction: 0.2 },
    }});
    assert.match(html, /^<details\b[^>]*id="legacy-result"/);
    assert.doesNotMatch(html.slice(0, html.indexOf(">")), /\bopen\b/, "The old terminal result starts collapsed");
    assert.match(html, /Прежний отчёт из терминала/);
    assert.match(html, /Прежняя проверка 80\/20/);
  },
  model_evidence() {
    setup();
    const legacy = job();
    assert.equal(get("modelSettingsState")(legacy).kind, "legacy");
    assert.equal(get("isStandardBaseline")(legacy), true);
    const standard = job({ model_config: { model: schema.model, parameters: { ...schema.defaults } } });
    assert.equal(get("modelSettingsState")(standard).kind, "standard");
    assert.equal(get("isStandardBaseline")(standard), true);
    const variant = job({ model_config: { model: schema.model, parameters: { ...schema.defaults, n_estimators: 160 } } });
    assert.equal(get("modelSettingsState")(variant).kind, "changed");
    assert.equal(get("modelSettingsState")(variant).changed, 1);
    assert.equal(get("isStandardBaseline")(variant), false, "The baseline action with changed parameters is a variant");
    assert.equal(get("isStandardBaseline")({ ...standard, action: "extra_trees" }), false);
    setup({ ...profile, model_options: {} });
    assert.equal(get("modelSettingsState")(standard).kind, "saved");
    assert.equal(get("isStandardBaseline")(standard), false, "Unknown defaults cannot prove a standard baseline");
  },
  baseline_reference() {
    const state = setup();
    const standard = job({ id: "standard", started_at: "2026-10-20T10:00:00Z", model_config: { model: schema.model, parameters: { ...schema.defaults } }, metric: { name: "Macro F1", value: 0.58, higher_is_better: true } });
    const variant = job({ id: "variant", started_at: "2026-10-20T11:00:00Z", model_config: { model: schema.model, parameters: { ...schema.defaults, n_estimators: 160 } }, metric: { name: "Macro F1", value: 0.62, higher_is_better: true } });
    const contender = job({ id: "contender", action: "extra_trees", started_at: "2026-10-20T12:00:00Z", metric: { name: "Macro F1", value: 0.6, higher_is_better: true } });
    state.jobs = [variant, standard, contender];
    const outcome = get("measuredComparison")(contender);
    assert.equal(outcome.tone, "better");
    assert.match(outcome.title, /Лучше.*baseline/);
    assert.match(outcome.note, /0,02/);
    state.jobs = [];
    assert.notEqual(get("measuredComparison")(variant).title, "Стартовая точка");
  },
  parameter_types() {
    setup();
    assert.deepEqual(plain(get("parseModelParams")(schema, rawValues())), {
      n_estimators: 160, max_depth: null, learning_rate: 0.2,
      class_weight: null, max_features: 0.5,
    });
    assert.equal(get("parseModelParams")(schema, { ...rawValues(), class_weight: '"balanced"' }).class_weight, "balanced");
    assert.equal(get("parseModelParams")(schema, { ...rawValues(), max_features: "1" }).max_features, 1);
    assert.equal(get("parseModelParams")(schema, { ...rawValues(), max_depth: "24" }).max_depth, 24);
  },
  parameter_errors() {
    setup();
    const invalid = [
      { n_estimators: "160.5" }, { n_estimators: "" }, { n_estimators: "9" },
      { n_estimators: "501" }, { n_estimators: "Infinity" }, { max_depth: "0" },
      { learning_rate: "NaN" }, { learning_rate: "1.1" }, { class_weight: '"unlisted"' },
      { max_features: "false" },
    ];
    for (const change of invalid)
      assert.throws(() => get("parseModelParams")(schema, { ...rawValues(), ...change }), JSON.stringify(change));
  },
  validation_inheritance() {
    const temporal = {
      ...profile, task_type: "regression",
      columns: [{ name: "Year_Factor", role: "feature", dtype: "numeric" }],
    };
    const state = setup(temporal);
    const reference = job({ validation: { strategy: "time", time_column: "Year_Factor" }, split: { strategy: "time", time_column: "Year_Factor" } });
    state.jobs = [reference]; state.selectedJob = reference.id; state.jobDetail = { job: reference };
    const edit = get("launchValidationConfig")("baseline", temporal, reference);
    assert.equal(edit.strategy, "time"); assert.equal(edit.time_column, "Year_Factor");
    assert.equal(get("launchValidationConfig")("extra_trees", temporal).strategy, "time");
    state.jobs = [job({ task_id: "other", validation: { strategy: "time", time_column: "Year_Factor" } })];
    state.selectedJob = null; state.jobDetail = null;
    assert.equal(get("launchValidationConfig")("extra_trees", temporal).strategy, "random");
    assert.equal(get("launchValidationConfig")("baseline", temporal).strategy, "random");
  },
  payload() {
    setup();
    const input = { task_id: "case", action: "baseline", author: "Дима", hypothesis: "Глубже", minutes: 20,
      validation: { strategy: "grouped" }, model_params: { n_estimators: 160, max_depth: null } };
    const offline = get("launchPayload")(input);
    assert.deepEqual(plain(offline.validation), input.validation);
    assert.deepEqual(plain(offline.model_params), input.model_params);
    for (const action of ["agent", "prepare"]) {
      const value = get("launchPayload")({ ...input, action });
      assert.equal("validation" in value, false, action);
      assert.equal("model_params" in value, false, action);
    }
  },
  automl_payload() {
    setup();
    const automatic = get("launchPayload")({
      task_id: "case", action: "automl", author: "Дима", hypothesis: "Подбор", minutes: 10,
      validation: { strategy: "auto" }, model_params: { n_estimators: 600 },
    });
    assert.equal(automatic.minutes, 10);
    assert.deepEqual(plain(automatic.validation), { strategy: "auto" });
    assert.equal("model_params" in automatic, false, "Search owns candidate parameters even if a previous recipe had edits");
    const explicit = get("launchPayload")({ ...automatic, validation: { strategy: "time", time_column: "Year_Factor" } });
    assert.deepEqual(plain(explicit.validation), { strategy: "time", time_column: "Year_Factor" });
    const qwen = get("launchPayload")({ ...automatic, action: "agent" });
    assert.equal("validation" in qwen, false);
    assert.equal("model_params" in qwen, false);
  },
  automl_validation() {
    const state = setup();
    const defaults = get("launchValidationConfig")("automl", profile);
    assert.equal(defaults.strategy, "auto");
    const automatic = get("validationFields")(profile, defaults, "automl");
    assert.match(automatic, /value="auto" checked/);
    const ordinary = get("validationFields")(profile, { strategy: "random" }, "baseline");
    assert.match(ordinary, /id="validation-auto-option" hidden/);
    assert.match(ordinary, /value="auto"\s+disabled/);
    const reference = job({ action: "automl", validation: { strategy: "auto" }, split: { strategy: "grouped" } });
    state.jobs = [reference]; state.selectedJob = reference.id; state.jobDetail = { job: reference };
    assert.equal(get("launchValidationConfig")("extra_trees", profile).strategy, "grouped", "A later manual comparison inherits the resolved search split");
    assert.equal(get("validationLabel")(reference), "По группам дублей");
    assert.equal(get("validationLabel")({ ...reference, split: null }), "Автоматический выбор проверки");
  },
  automl_results() {
    setup();
    const search = {
      status: "time_limit", planned: 4, completed: 2, best_candidate: "earlier",
      validation_reason: '<script>reason()</script>',
      candidates: [
        { id: "earlier", label: 'Первый <img src=x>', model: "RandomForestClassifier", status: "completed", metric: 0.6, selected: true, elapsed_seconds: 10, error: null },
        { id: "later", label: "Поздний", model: "ExtraTreesClassifier", status: "completed", metric: 0.59, selected: false, elapsed_seconds: 12, error: null },
        { id: "timeout", label: "Не успел", model: "ExtraTreesClassifier", status: "timed_out", metric: null, selected: false, elapsed_seconds: 60, error: '<svg/onload=evil()>' },
        { id: "skipped", label: "Не запускался", model: "ExtraTreesClassifier", status: "skipped", metric: null, selected: false, elapsed_seconds: 0, error: null },
      ],
    };
    const current = job({ action: "automl", search });
    const html = get("autoSearchView")(current);
    assert.match(html, /Лимит времени · лучший сохранён/);
    assert.match(html, /Оценено 2 \/ 4 вариантов/);
    assert.equal((html.match(/search-selected/g) || []).length, 1);
    assert.match(html, /search-selected[^]*Первый &lt;img/);
    assert.match(html, /Пропущен/);
    assert.doesNotMatch(html, /<script>|<img|<svg\/onload/);
    assert.match(html, /&lt;svg\/onload/);
    const unsupported = get("autoSearchView")({ ...current, search: { ...search, best_candidate: "unknown" } });
    assert.doesNotMatch(unsupported, /search-selected|лучший сохранён/);
    assert.equal(get("autoSearchView")({ ...current, action: "agent" }), "");
    const grouped = get("comparisonGroups")([current, job({ id: "manual", metric: { name: "Macro F1", value: 0.5, higher_is_better: true } })]);
    assert.equal(grouped.length, 1);
    assert.equal(grouped[0].best.id, current.id, "The measured search result participates in existing comparison groups");
    cases.expanded_search();
  },
  expanded_search() {
    const actions = ["baseline", "extra_trees", "catboost", "lightgbm", "linear"];
    setup({
      ...profile, task_type: "regression", metric: { name: "R2", higher_is_better: true },
      model_options: Object.fromEntries(actions.map(action => [action, schema])),
    });
    assert.deepEqual(
      plain(get("launchRecipeOptions")(get("currentProfile")()).map(row => row[0])),
      ["automl", ...actions, "agent"],
      "All five advertised model recipes must be available in the launch dialog",
    );
    for (const action of ["catboost", "lightgbm", "linear"]) {
      const payload = plain(get("launchPayload")({
        task_id: "case", action, author: "", hypothesis: "", minutes: 20,
        validation: { strategy: "time", time_column: "x" }, model_params: { depth: 4 },
      }));
      assert.equal(payload.action, action);
      assert.deepEqual(payload.validation, { strategy: "time", time_column: "x" });
      assert.deepEqual(payload.model_params, { depth: 4 });
    }
    const candidates = Array.from({ length: 22 }, (_, index) => ({
      id: `candidate-${index}`, label: `candidate-${index}`, model: "Ridge", phase: "selection",
      status: "completed", metric: null, verification_metric: null, selection_metric: -2,
      selected: false, elapsed_seconds: 1, error: null,
    }));
    Object.assign(candidates[0], {
      metric: -3, verification_metric: -3, selected: true, phase: "seed",
      resources: { available: true, peak_rss_bytes: 1024 * 1024 },
    });
    Object.assign(candidates[1], {
      selection_metric: 0.99,
      feature_config: { missing_indicators: true, rare_min_count: 0, date_columns: ["date"], text_columns: [], signed_log: false },
    });
    Object.assign(candidates[2], {
      status: "failed", selection_metric: null, rejection_reason: "memory_limit",
      error: '<img src=x onerror="evil()">', resources: { available: false, peak_rss_bytes: null },
    });
    const current = job({
      action: "automl", metric: { name: "R2", value: -3, higher_is_better: true },
      search: { status: "completed", phase: "finished", planned: 22, completed: 21,
        best_candidate: "candidate-0", candidates },
    });
    const html = get("autoSearchView")(current);
    assert.match(html, /candidate-21/, "All 22 scheduled recipes including metric-aware candidates must remain visible");
    assert.match(html, /R2/);
    assert.match(html, /class="metric-cell">-3</);
    assert.doesNotMatch(html, /Macro F1|RMSE/, "Metric direction must not imply the task type");
    assert.match(html, /candidate-1<[^]*?metric-cell">0,99<\/td><td class="metric-cell">—/);
    assert.match(html, /Подбор завершён/);
    assert.equal((html.match(/search-selected/g) || []).length, 1);
    assert.match(html, /1 МБ/);
    assert.match(html, /Не измерялось/);
    assert.match(html, /Лимит памяти/);
    assert.match(html, /Индикаторы пропусков/);
    assert.match(html, /Даты/);
    assert.match(html, /&lt;img src=x onerror=&quot;evil\(\)&quot;&gt;/);
    assert.doesNotMatch(html, /<img|onerror="evil\(\)"/);
    current.search.best_candidate = "candidate-1";
    candidates[0].selected = false;
    candidates[1].selected = true;
    const invalidWinner = get("autoSearchView")(current);
    assert.doesNotMatch(invalidWinner, /search-selected|Лучший вариант сохранён|>Выбран</,
      "An inner-only metric cannot be displayed as the verified winner even if selected is claimed");
  },
  automl_progress() {
    setup();
    const current = job({ action: "automl", status: "running", metric: null, output_valid: null,
      search: { status: "running", planned: 4, completed: 0, best_candidate: null,
        validation_reason: "Одинаковые строки остаются вместе", candidates: [
          { id: "a", status: "running", metric: null, selected: false },
          { id: "b", status: "skipped", metric: null, selected: false },
          { id: "c", status: "skipped", metric: null, selected: false },
          { id: "d", status: "skipped", metric: null, selected: false },
        ] },
    });
    const html = get("autoSearchView")(current);
    assert.match(html, /Оценено 0 \/ 4 вариантов/);
    assert.match(html, /value="0"/);
    assert.match(html, /Выполняется/);
    assert.match(html, /В очереди/);
    assert.doesNotMatch(html, /Пропущен|Выбран|лучший сохранён/);
    current.search.candidates[0] = { id: "a", status: "completed", metric: 0.5, selected: true };
    current.search.best_candidate = "a";
    assert.match(get("autoSearchView")(current), /Пока лучший/);
    assert.doesNotMatch(get("autoSearchView")(current), />Выбран</);
    current.search.status = "failed";
    assert.match(get("autoSearchView")(current), /завершился с ошибкой/);
    assert.doesNotMatch(get("autoSearchView")(current), /В очереди|>Выбран</);
  },
  diagnostic_orientation() {
    setup();
    const current = diagnosticJob(), evidence = get("classificationDiagnosticEvidence")(current);
    assert.deepEqual(plain(evidence.rows.map(row => row.label)), ["Worms", "Other", "Normal"], "Weakest measured classes come first");
    assert.deepEqual(plain(evidence.confusions[0]), { actual: "Normal", prediction: "Worms", count: 10, fraction: 0.1 });
    const html = get("classificationDiagnosticView")(current);
    assert.match(html, /Где ошибается модель/);
    assert.ok(html.indexOf('data-diagnostic-class="Worms"') < html.indexOf('data-diagnostic-class="Normal"'));
    assert.match(html, /Истинный класс: Worms; предсказано: Normal; 2 из 4 \(50%\)/);
    assert.match(html, /style="--cell-strength:0\.5000" title="Истинный класс: Worms; предсказано: Normal/);
    assert.match(html, /style="--cell-strength:0\.1000" title="Истинный класс: Normal; предсказано: Worms/);
    assert.match(html, /Прогнозов: 10/);
    assert.match(html, /Строки — истинный класс, столбцы — прогноз/);
    assert.match(get("jobMetricView")(current), /Разбор ошибок классификации/, "The panel is part of the selected job's metric view");
  },
  diagnostic_perfect_and_empty_class() {
    setup();
    const current = diagnosticJob({
      metric: { name: "Macro F1", value: 1, higher_is_better: true },
      split: { validation_rows: 3 },
      classification_diagnostics: {
        version: 1, kind: "classification", labels: ["A", "B"], rows: 3,
        confusion_matrix: [[2, 0], [0, 1]],
        per_class: [
          { precision: 1, recall: 1, f1: 1, support: 2, predicted: 2 },
          { precision: 1, recall: 1, f1: 1, support: 1, predicted: 1 },
        ], accuracy: 1, macro_f1: 1,
      },
    });
    assert.match(get("classificationDiagnosticView")(current), /Все проверочные примеры распознаны верно/);
    assert.equal(get("classificationDiagnosticEvidence")(current).confusions.length, 0);
    current.classification_diagnostics.confusion_matrix = [[2, 1], [0, 0]];
    current.classification_diagnostics.per_class = [
      { precision: 1, recall: 2 / 3, f1: 0.8, support: 3, predicted: 2 },
      { precision: 0, recall: 0, f1: 0, support: 0, predicted: 1 },
    ];
    current.classification_diagnostics.accuracy = 2 / 3;
    current.classification_diagnostics.macro_f1 = 0.4;
    current.metric.value = 0.4;
    const evidence = get("classificationDiagnosticEvidence")(current);
    assert.deepEqual(plain(evidence.rows.map(row => row.label)), ["A", "B"], "An unobserved class is not presented as the weakest measured class");
    const html = get("classificationDiagnosticView")(current);
    assert.match(html, /Нет примеров в проверке/);
    assert.doesNotMatch(html, /NaN|Infinity|Все проверочные примеры распознаны верно/);
  },
  diagnostic_boundaries() {
    setup();
    const view = get("classificationDiagnosticView");
    assert.equal(view(job()), "", "Old jobs without predictions do not invent diagnostics");
    assert.equal(view(diagnosticJob({ metric: { name: "RMSE", value: 1, higher_is_better: false } })), "");
    assert.equal(view(diagnosticJob({ status: "running" })), "");
    for (const change of [
      current => { current.classification_diagnostics.rows = 111; },
      current => { current.classification_diagnostics.confusion_matrix[0][0] = true; },
      current => { current.classification_diagnostics.confusion_matrix[0][0] = -1; },
      current => { current.classification_diagnostics.per_class[0].f1 = NaN; },
      current => { current.classification_diagnostics.per_class[0].f1 = 1; },
      current => { current.classification_diagnostics.labels[0] = current.classification_diagnostics.labels[1]; },
      current => { current.split.validation_rows = 120; },
      current => { current.metric.value = 0.999; },
    ]) {
      const current = diagnosticJob(); change(current);
      assert.equal(view(current), "", "Malformed or mismatched evidence must not produce a misleading chart");
    }
  },
  diagnostic_escaping_and_provenance() {
    setup();
    const current = diagnosticJob({ source: "imported" });
    current.classification_diagnostics.labels[1] = '<img src=x onerror="evil()">';
    const html = get("classificationDiagnosticView")(current);
    assert.match(html, /&lt;img src=x onerror=&quot;evil\(\)&quot;&gt;/);
    assert.doesNotMatch(html, /<img|onerror="evil\(\)"/);
    assert.match(html, /Отчёт участника · не проверен локально/);
    assert.doesNotMatch(html, /На проверочных строках этого запуска/);
    current.source = "local";
    assert.match(get("classificationDiagnosticView")(current), /На проверочных строках этого запуска/);
    assert.doesNotMatch(get("classificationDiagnosticView")(current), /не проверен локально/);
  },
  research_stages_and_provenance() {
    setup();
    const summary = {
      protocol_version: 3, stage: "outer_validation",
      stages: [
        { id: "inner_research", status: "completed" },
        { id: "candidate_frozen", status: "completed" },
        { id: "outer_validation", status: "running" },
        { id: "full_refit", status: "skipped", reason: '<img src=x onerror="evil()">' },
      ],
      evaluations: {
        selection: { scope: "inner_selection", metric: { name: "R2", value: -0.25, higher_is_better: true }, split: { strategy: "time", train_rows: 40, validation_rows: 10 } },
        final_check: { baseline: { name: "R2", value: -0.5, higher_is_better: true }, candidate: { name: "R2", value: -0.75, higher_is_better: true }, selected: "baseline" },
      },
      frozen_candidate: { code_hash: "a".repeat(64), recipe_hash: "b".repeat(64) },
    };
    const html = get("researchView")(summary, { status: "fallback" });
    assert.match(html, /aria-current="step"/);
    assert.equal((html.match(/class="research-stage /g) || []).length, 6);
    assert.match(html, /Внутренний подбор/);
    assert.match(html, /Внешняя приёмка/);
    assert.match(html, /Сохранён стартовый рецепт/);
    assert.match(html, /не является независимой оценкой/);
    assert.match(html, /резервный результат/);
    assert.match(html, /-0,25/);
    assert.match(html, /Условия этапа не выполнены/);
    assert.doesNotMatch(html, /<img|onerror="evil/);
    const imported = get("researchView")(summary, { imported: true });
    assert.match(imported, /отчёт участника, локальная проверка не выполнялась/);
    const legacy = get("researchView")({ protocol_version: 2 });
    assert.match(legacy, /Прежний протокол v2/);
    assert.doesNotMatch(legacy, /Внутренний подбор|Протокол v3/);
    assert.match(get("usageView")({ state: "unknown", total_tokens: 0 }), /неизвестно/);
  },
  research_not_better_is_normal() {
    const html = get("eventView")({ kind: "experiment", step: 1, tool_index: 1,
      result: { exit_code: 0, elapsed_seconds: 1.2, selection: {
        verified: true, accepted: false, reason: "not_better", hypothesis: "<script>evil()</script>",
        scope: "inner_selection", recipe_summary: { family: "Ridge" },
        metric: { name: "RMSE", value: 4.2, higher_is_better: false },
      } } }, 0);
    assert.match(html, /timeline-event success/);
    assert.match(html, /улучшения нет/);
    assert.match(html, /RMSE/);
    assert.match(html, /Внутренняя проверка · подбор рецепта/);
    assert.match(html, /Ridge/);
    assert.match(html, /&lt;script&gt;/);
    assert.doesNotMatch(html, /timeline-event error|<script>/);
  },
  research_fallback_reasons_and_private_paths() {
    const labels = {
      context_budget: /лимит контекста модели/,
      inner_validation_unavailable: /внутреннюю проверку/,
      refit_failed: /обучение на всех данных/,
    };
    for (const [reason, label] of Object.entries(labels)) {
      assert.match(get("researchView")({ protocol_version: 2 }, { status: "fallback", reason }), label);
      const current = get("researchView")({ protocol_version: 3, stage: "inner_research",
        stages: [{ id: "inner_research", status: "failed", reason: "/workspace/private/run.json" }] },
        { status: "fallback", reason });
      assert.match(current, label);
      assert.match(current, /Не выполнялся/);
      assert.doesNotMatch(current, /workspace|private|run\.json|Ожидает/);
    }
    const unknown = get("researchView")({ protocol_version: 2 }, { status: "fallback", reason: "/workspace/private/error.log" });
    assert.doesNotMatch(unknown, /workspace|private|error\.log/);
    const event = get("eventView")({ kind: "experiment", result: { exit_code: 0, selection: {
      verified: true, scope: "inner_selection", recipe_summary: { family: "/workspace/private/model.py" },
    } } }, 0);
    assert.match(event, /Пользовательский рецепт/);
    assert.doesNotMatch(event, /workspace|private|model\.py/);
    const interrupted = get("researchView")({ protocol_version: 3, stage: "inner_research",
      stages: [{ id: "inner_research", status: "running" }] }, { status: "interrupted" });
    assert.match(interrupted, /Остановлен/);
    assert.doesNotMatch(interrupted, /spinner|Выполняется|Ожидает/);
  },
  api_required_fields() {
    const state = setup();
    state.session = { api: { ready: false } };
    get("showHelp")();
    assert.match(element.innerHTML, /OPENAI_API_KEY/);
    assert.match(element.innerHTML, /OPENAI_BASE_URL/);
    assert.doesNotMatch(element.innerHTML, /OPENAI_MODEL/);
    state.session.api = { ready: true, missing: [], model_configured: true };
    get("showHelp")();
    assert.match(element.innerHTML, /Настройки API заполнены/);
    assert.doesNotMatch(element.innerHTML, /OPENAI_API_KEY|OPENAI_BASE_URL|OPENAI_MODEL/);
  },
  escaping() {
    setup();
    const malicious = job({ model_config: { model: '<img src=x onerror="evil()">', parameters: { '<script>evil()</script>': '"<svg/onload=evil()>"' } } });
    const html = get("modelParameterView")(malicious);
    assert.doesNotMatch(html, /<img|<script>|<svg\/onload/);
    assert.match(html, /&lt;/, "Unknown persisted parameter names must be text");
    const missing = get("modelParameterView")(job());
    assert.doesNotMatch(missing, /n_estimators|max_depth|balanced/, "No current defaults may be fabricated as historical evidence");
  },
};

(async () => {
  await new Promise(resolve => setImmediate(resolve));
  const name = process.argv[2];
  assert.equal(typeof cases[name], "function", `Unknown frontend case ${name}`);
  cases[name]();
  console.log(`Frontend ${name}: OK`);
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
