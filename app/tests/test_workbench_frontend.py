"""Behavioral checks for workbench readiness, reviewed plans and honest telemetry."""
from pathlib import Path
import shutil
import subprocess
import unittest

from app.tests.test_native_frontend import SCRIPT as NATIVE_SCRIPT


ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
SCRIPT = NATIVE_SCRIPT[:NATIVE_SCRIPT.index("async function main() {")] + r'''
const presets = {presets: [
  {id: "smoke", name: "Короткая проверка", description: "5 минут", actions: ["automl"], budgets_minutes: [5]},
  {id: "models", name: "Сравнение моделей", description: "5 семейств", actions: ["baseline", "extra_trees", "catboost", "lightgbm", "linear"], budgets_minutes: [30]},
  {id: "day", name: "Дневная серия", description: "30/90/175", actions: ["automl"], budgets_minutes: [30, 90, 175]},
  {id: "custom", name: "Свой план", description: "12 заданий", actions: ["automl"], budgets_minutes: []},
]};
function campaignSetup() {
  const state = setup(); state.view = "campaigns";
  state.campaignPresets = presets; state.campaignActive = null; state.campaigns = [];
  state.campaignDraft = get("newCampaignDraft")(); state.campaignRevision = 0;
  state.campaignPreview = null; state.campaignPreviewBody = null; state.campaignPreviewRevision = null;
  state.campaignPreviewBusy = false; state.campaignStartBusy = false; state.campaignPollBusy = false;
  state.campaignError = ""; state.campaignControlBusy = null;
  return state;
}
function validPreview(state) {
  state.campaignPreview = {ready: true, errors: [], warnings: [], source_seal: {valid: true, sha: "c".repeat(40)}, runtime_seal: {valid: true, python_version: "3.12.2"}, plan_sha256: "d".repeat(64)};
  state.campaignPreviewBody = plain(get("campaignBody")());
  state.campaignPreviewRevision = state.campaignRevision;
}
async function main() {
  await new Promise((resolve) => setImmediate(resolve));
  const which = process.argv[2];
  if (which === "presets_payload") {
    const state = campaignSetup();
    assert.deepEqual(state.campaignDraft.steps.map((step) => step.minutes), [30,30,90,90,175,175]);
    assert.deepEqual(state.campaignDraft.steps.map((step) => step.task_id), ["reg","cls","reg","cls","reg","cls"]);
    assert.equal(state.campaignDraft.steps.reduce((total, step) => total + step.minutes, 0), 590);
    const builtin = plain(get("campaignBody")());
    assert.equal(builtin.preset, "day"); assert.equal("steps" in builtin, false);
    get("selectCampaignPreset")("models"); assert.equal(state.campaignDraft.steps.length, 10);
    get("selectCampaignPreset")("custom"); assert.equal(state.campaignDraft.steps.length, 10, "Custom preserves the editable plan");
    assert.equal(get("campaignBody")().steps.length, 10);
    state.campaignDraft.steps[0].action = "agent";
    assert.throws(() => get("campaignBody")(), /известный метод/);
  } else if (which === "draft_validation") {
    const state = campaignSetup(); get("selectCampaignPreset")("custom");
    for (const minutes of [0, 176, 1.5, NaN]) {
      state.campaignDraft.steps[0].minutes = minutes; assert.throws(() => get("campaignBody")(), /целое число/);
    }
    state.campaignDraft.steps[0].minutes = 30;
    state.campaignDraft.steps = Array.from({length: 13}, () => structuredClone(state.campaignDraft.steps[0]));
    assert.throws(() => get("campaignBody")(), /от 1 до 12/);
    state.campaignDraft.steps.length = 1; state.campaignDraft.max_hours = 13;
    assert.throws(() => get("campaignBody")(), /12 часов/);
    state.campaignDraft.max_hours = 12; state.campaignDraft.steps[0].task_id = "missing";
    assert.throws(() => get("campaignBody")(), /готовые задачи/);
  } else if (which === "edit_invalidates") {
    const state = campaignSetup(); validPreview(state);
    assert.equal(get("campaignLaunchBlock")(), null);
    get("moveCampaignStep")(0, 1);
    assert.equal(state.campaignDraft.preset, "custom"); assert.equal(state.campaignDraft.steps[0].task_id, "cls");
    assert.equal(state.campaignPreview, null); assert.match(get("campaignLaunchBlock")(), /проверьте план/);
    assert.equal(state.campaignEditorOpen, true, "Editing keeps the advanced editor open");
    validPreview(state); get("changeCampaignField")(0, "minutes", "90");
    assert.equal(state.campaignDraft.steps[0].minutes, 90); assert.equal(state.campaignPreview, null);
    get("changeCampaignField")(0, "action", "baseline");
    assert.equal(state.campaignDraft.steps[0].minutes, 30);
    assert.equal(state.campaignDraft.steps[0].validation.strategy, "random");
    validPreview(state); state.campaignPreview.runtime_seal.valid = false;
    assert.match(get("campaignLaunchBlock")(), /Среда Python/);
    assert.match(get("campaignSealsHTML")(state.campaignPreview), /Среда Python изменилась/);
  } else if (which === "frozen_start") {
    const state = campaignSetup(); validPreview(state); state.view = "data";
    const frozen = plain(state.campaignPreviewBody);
    responder = (url) => {assert.equal(url, "/api/campaigns"); return {campaign: {id, status: "running"}};};
    await get("startCampaign")();
    const request = calls.find((call) => call.options?.method === "POST");
    assert.deepEqual(JSON.parse(request.options.body), {...frozen, preview_token: "d".repeat(64)});
    assert.equal(request.options.headers["X-Alfa-Token"], "test-token");
    assert.equal(state.campaignActive.id, id); assert.equal(state.campaignPreview, null);
  } else if (which === "stale_preview") {
    const state = campaignSetup(); let finish;
    responder = () => new Promise((resolve) => {finish = resolve;});
    const request = get("previewCampaign")(); await new Promise((resolve) => setImmediate(resolve));
    get("changeCampaignField")(0, "minutes", "60");
    finish({ready: true, steps: [], source_seal: {valid: true}, plan_sha256: "d".repeat(64)});
    await request;
    assert.equal(state.campaignPreview, null); assert.equal(state.campaignPreviewBody, null);
    assert.equal(state.campaignDraft.steps[0].minutes, 60);
  } else if (which === "preview_failure") {
    const state = campaignSetup(), before = state.campaignDraft.steps.length;
    responder = () => ({ready: false, errors: ["Source unavailable"], warnings: [], steps: []});
    await get("previewCampaign")();
    assert.equal(state.campaignDraft.steps.length, before, "A failed source check must not erase the user's plan");
    assert.match(get("campaignLaunchBlock")(), /Исправьте/);
    responder = () => {throw new Error("Offline now");};
    await get("previewCampaign")();
    assert.match(state.campaignError, /Сервер вернул непонятный ответ/);
    assert.match(element("#campaign-editor").innerHTML, /Сервер вернул непонятный ответ/);
  } else if (which === "paused_lane") {
    const state = campaignSetup(); validPreview(state);
    const button = element("ordinary"); button.disabled = false; launchButtons.push(button);
    get("setCampaignActive")({id, status: "paused"});
    assert.equal(button.disabled, true);
    assert.match(get("campaignLaunchBlock")(), /занимает очередь/);
    assert.match(get("nativeLaunchBlock")(), /Серия тестов занимает очередь/);
    const generation = state.launchGeneration; await get("showLaunch")("baseline");
    assert.equal(state.launchGeneration, generation);
    get("setCampaignActive")(null); assert.equal(button.disabled, false);
  } else if (which === "controls") {
    campaignSetup(); const campaign = {id, name: "Plan", status: "running", steps: [], wall_budget_seconds: 43200};
    let html = get("campaignDetailHTML")(campaign);
    assert.match(html, /data-campaign-control="pause"/); assert.doesNotMatch(html, /data-campaign-control="resume"/);
    html = get("campaignDetailHTML")({...campaign, status: "pausing"});
    assert.doesNotMatch(html, /data-campaign-control="pause"|data-campaign-control="resume"/);
    assert.match(html, /Текущее задание продолжает работу/);
    html = get("campaignDetailHTML")({...campaign, status: "paused", steps: [{status: "pending", action: "automl", task_id: "reg", minutes: 5}]});
    assert.match(html, /data-campaign-control="resume"/); assert.match(html, /только оставшиеся задания/);
    html = get("campaignDetailHTML")({...campaign, status: "completed"});
    assert.doesNotMatch(html, /data-campaign-control=/);
    html = get("campaignDetailHTML")({...campaign, status: "cleanup_pending"});
    assert.match(html, /Повторить остановку/); assert.doesNotMatch(html, /resume/);
    assert.equal(get("campaignCompleted")({steps: [{status: "completed"}, {status: "cancelled"}, {status: "pending"}]}), 1,
      "Unstarted cancelled steps must not inflate the completed task count");
  } else if (which === "ordinary_cleanup") {
    const state = campaignSetup(), job = {id, task_id: "reg", action: "baseline", status: "cleanup_pending", stage: "cleanup", source: "local"};
    state.session.active_job = job;
    assert.equal(get("runningJob")(job), true);
    assert.match(get("statusBadge")(job.status), /Остановка требует проверки/);
    assert.match(get("campaignLaunchBlock")(), /отдельный эксперимент/);
    assert.match(get("nativeLaunchBlock")(), /другой запуск/);
    assert.match(get("overviewActivity")({active_job: job}), /Остановка требует проверки/);
    state.view = "runs"; state.task = "reg"; state.selectedJob = id;
    responder = () => ({job, stdout: "", stderr: ""});
    await get("loadJobDetail")(id);
    assert.match(element("#job-detail").innerHTML, /data-action="stop-job">Повторить остановку/);
    assert.equal(state.jobDetail.job.status, "cleanup_pending");
  } else if (which === "interrupted_resume") {
    campaignSetup();
    const campaign = {id, status: "interrupted", deadline_at: "2026-10-24T08:00:00Z", steps: [
      {status: "interrupted", task_id: "reg", action: "automl", minutes: 30},
      {status: "pending", task_id: "cls", action: "automl", minutes: 30},
    ]};
    let html = get("campaignDetailHTML")(campaign);
    assert.match(html, /data-campaign-control="resume"/);
    assert.match(html, /прежнее обучение не повторяется/);
    assert.match(html, /первоначальный срок серии/);
    assert.match(html, /время после закрытия приложения его не продлевают/);
    html = get("campaignDetailHTML")({...campaign, steps: campaign.steps.slice(0, 1)});
    assert.doesNotMatch(html, /data-campaign-control="resume"/);
    for (const status of ["completed", "stopped", "failed", "cleanup_pending"])
      assert.doesNotMatch(get("campaignDetailHTML")({...campaign, status}), /data-campaign-control="resume"/);
  } else if (which === "comparison_scope") {
    campaignSetup();
    const group = {task_id: "reg", dataset_fingerprint: "e".repeat(64), comparison_key: "f".repeat(64),
      metric_name: "RMSE", higher_is_better: false, best_job_id: otherId, split: {strategy: "random"},
      jobs: [{job_id: id, action: "automl", minutes: 30, value: 10}, {job_id: otherId, action: "automl", minutes: 90, value: 9}]};
    const html = get("campaignComparisonsHTML")([group]);
    assert.match(html, /Лучший в этой локальной группе/); assert.match(html, /<polyline/);
    assert.match(html, /Лимит времени, а не фактически потраченное/);
    assert.match(get("campaignComparisonsHTML")([{...group, comparison_key: null}]), /Сравнение появится/);
    assert.doesNotMatch(get("campaignComparisonsHTML")([{...group, jobs: [{job_id: id, action: "automl", minutes: 30, value: NaN}]}]), /★|<polyline/);
    assert.doesNotMatch(get("campaignCurve")(group, [{job_id: id, action: "baseline", minutes: 30, value: 1}]), /<polyline/);
    const single = get("campaignComparisonsHTML")([{...group, comparison_available: false, jobs: [group.jobs[0]], best_job_id: id}]);
    assert.match(single, /Пока один результат/);
    assert.doesNotMatch(single, /Лучший в этой локальной группе|★|<polyline/);
    const detail = {id, steps: [{id: "s1", job_id: id, action: "automl"}, {id: "s2", job_id: otherId, action: "automl"}]};
    const report = {campaign: {id}, groups: [{...group, best: {job_id: otherId}, budget_curve: [
      {step_id: "s1", job_id: id, minutes: 30, metric: {value: 10}},
      {step_id: "s2", job_id: otherId, minutes: 90, metric: {value: 9}},
      {step_id: "unbound", job_id: "forged", minutes: 175, metric: {value: 0}},
    ]}]};
    const linked = get("campaignReportGroups")(report, detail);
    assert.equal(linked[0].jobs.length, 2, "UI comparisons use the exported report and retain step/job binding");
    assert.equal(linked[0].best_job_id, otherId);
    assert.equal(get("campaignReportGroups")({...report, campaign: {id: otherId}}, detail).length, 0);
  } else if (which === "telemetry") {
    campaignSetup();
    const html = get("systemHTML")({cpu: {name: '<img src=x onerror="evil()">', percent: null}, memory: {percent: 0, total_bytes: 32 * 1024**3, available_bytes: 32 * 1024**3}, disk: {}, gpu: {reason: '<script>evil()</script>', devices: []}, sleep: {supported: true, requested: true, active: false, reason: "No permission"}});
    assert.match(html, /Нет измерения/); assert.match(html, /0 %/);
    assert.match(html, /GPU для обучения не включён/); assert.match(html, /всего компьютера/);
    assert.match(html, /запрос против сна не подтверждён/);
    assert.doesNotMatch(html, /<img|<script>/);
    assert.doesNotMatch(get("systemHTML")(null), /0 %/);
  } else if (which === "stale_detail") {
    const state = campaignSetup(); let finish;
    state.campaignSelected = id; state.campaignDetail = {id: otherId};
    responder = (url) => url.endsWith("/report.json") ? {campaign: {id}, groups: []}
      : new Promise((resolve) => {finish = resolve;});
    const request = get("loadCampaignDetail")(id, 20); await new Promise((resolve) => setImmediate(resolve));
    state.campaignSelected = otherId; finish({campaign: {id, status: "completed", steps: []}}); await request;
    assert.equal(state.campaignDetail.id, otherId);
  } else if (which === "poll_scope") {
    const state = campaignSetup(); context.document.hidden = false;
    responder = (url) => url === "/api/campaigns" ? {campaigns: [], active_campaign: null} : {token: "test-token", active_job: null, active_queue: null, active_campaign: null};
    state.view = "data"; await get("pollWorkbench")(); assert.equal(calls.length, 0);
    state.view = "campaigns"; await get("pollWorkbench")();
    assert.deepEqual(calls.map((call) => call.url).sort(), ["/api/campaigns", "/api/session"]);
    assert.equal(calls.some((call) => call.url.includes("preview") || call.options?.method), false);
  } else if (which === "overview_empty") {
    const state = campaignSetup(); state.view = "overview"; state.tasks = []; state.task = null;
    responder = (url) => url === "/api/dashboard" ? {tasks: [], recent_jobs: [], active_job: null, active_queue: null, active_campaign: null} : {};
    await get("loadOverview")(20);
    assert.match(element("#main").innerHTML, /Обзор/); assert.match(element("#main").innerHTML, /Загрузить публичные данные/);
    assert.equal(calls.some((call) => /\/workflow$|\/profile$/.test(call.url)), false);
  } else throw new Error("Unknown workbench check " + which);
}
main().catch((error) => {console.error(error.stack || error); process.exitCode = 1;});
'''


@unittest.skipUnless(NODE, "Workbench checks require Node.js")
class WorkbenchFrontendTests(unittest.TestCase):
    def test_workbench_behavior_and_boundaries(self):
        for case in ("presets_payload", "draft_validation", "edit_invalidates", "frozen_start", "stale_preview",
                     "preview_failure", "paused_lane", "controls", "ordinary_cleanup", "interrupted_resume", "comparison_scope", "telemetry",
                     "stale_detail", "poll_scope", "overview_empty"):
            with self.subTest(case=case):
                result = subprocess.run([NODE, "-e", SCRIPT, str(ROOT / "app/static/app.js"), case],
                                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_overview_and_all_workflows_are_accessible_from_navigation(self):
        html = (ROOT / "app/static/index.html").read_text(encoding="utf-8")
        for route in ("overview", "data", "runs", "result", "campaigns", "native"):
            self.assertIn(f'data-view="{route}"', html)
        self.assertIn('href="#overview"', html)


if __name__ == "__main__":
    unittest.main()
