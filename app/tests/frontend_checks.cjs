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
