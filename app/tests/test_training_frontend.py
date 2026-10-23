"""Model-free behavioral checks for reviewed training studies and local artifacts."""
from pathlib import Path
import shutil
import subprocess
import unittest

from app.tests.test_native_frontend import SCRIPT as NATIVE_SCRIPT


ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
SCRIPT = NATIVE_SCRIPT[:NATIVE_SCRIPT.index("async function main() {")] + r'''
const trainingPresets = {presets: [
  {id: "day", name: "Дневная серия", actions: ["automl"], budgets_minutes: [30,90,175]},
  {id: "custom", name: "Свой план", actions: ["automl"], budgets_minutes: []},
  {id: "lightgbm_trees", name: "Сравнение обучения LightGBM", actions: ["lightgbm"], budgets_minutes: [30]},
]};
function sourceJob(changes = {}) {
  return {id, task_id: "cls", source: "local", status: "completed", action: "automl", output_valid: true,
    started_at: "2026-10-23T10:00:00Z", hypothesis: "Лучший LightGBM",
    dataset_fingerprint: "d".repeat(64), validation: {strategy: "grouped"},
    model_config: {version: 1, model: "LGBMClassifier", parameters: {n_estimators: 500, num_leaves: 63, class_weight: "balanced"}},
    feature_config: {missing_indicators: true, rare_min_count: 10, signed_log: true, date_columns: [], text_columns: []},
    ...changes};
}
function studySetup() {
  const state = setup(); state.view = "campaigns"; state.task = "cls";
  state.campaignPresets = trainingPresets; state.campaignSourceJobs = [sourceJob()];
  state.campaigns = []; state.campaignActive = null; state.jobs = []; state.experimentReports = [];
  state.campaignDraft = get("newCampaignDraft")(); state.campaignRevision = 0;
  state.campaignPreview = null; state.campaignPreviewBody = null; state.campaignPreviewRevision = null;
  state.campaignPreviewBusy = false; state.campaignStartBusy = false; state.campaignError = "";
  state.campaignStudyRequest = null; state.launchTrainingDevice = "cpu";
  state.trainingDevicesBusy = false; state.trainingDevices = null;
  get("selectCampaignPreset")("lightgbm_trees");
  return state;
}
async function main() {
  await new Promise((resolve) => setImmediate(resolve));
  const which = process.argv[2];
  if (which === "study_payload") {
    const state = studySetup(), body = plain(get("campaignBody")());
    assert.deepEqual(body, {preset: "lightgbm_trees", task_ids: ["cls"], name: "Сравнение обучения LightGBM", max_hours: 12, minutes_per_step: 30});
    assert.deepEqual(plain(state.campaignDraft.steps.map((step) => step.model_params.n_estimators)), [500,750,1000]);
    assert.equal(state.campaignDraft.steps[0].is_control, true);
    assert.match(get("campaignEditorHTML")(), /Базовый рецепт.*500 \/ 750 \/ 1000/);
    assert.match(get("campaignEditorHTML")(), /не считается точной копией/);
    state.campaignDraft.task_ids = ["cls", "reg"]; assert.throws(() => get("campaignBody")(), /одну готовую задачу/);
    state.campaignDraft.task_ids = ["cls"]; state.campaignDraft.minutes_per_step = 1.5;
    assert.throws(() => get("campaignBody")(), /целое число/);
  } else if (which === "inherit_source") {
    const state = studySetup(); get("updateLightgbmStudy")(id, 60);
    assert.equal(get("campaignBody")().source_job_id, id);
    assert.equal(get("campaignBody")().minutes_per_step, 60);
    assert.deepEqual(plain(state.campaignDraft.steps.map((step) => step.model_params.n_estimators)), [500,750,1000]);
    for (const step of state.campaignDraft.steps) {
      assert.equal(step.validation.strategy, "grouped");
      assert.equal(step.feature_config.signed_log, true);
      assert.equal(step.model_params.class_weight, "balanced");
      assert.equal(step.minutes, 60);
    }
    state.campaignSourceJobs = [sourceJob({model_config: {...sourceJob().model_config, parameters: {n_estimators: 1500}}})];
    get("updateLightgbmStudy")(id, 30);
    assert.deepEqual(plain(state.campaignDraft.steps.map((step) => step.model_params.n_estimators)), [1500,2000]);
    state.campaignSourceJobs = [sourceJob({model_config: {...sourceJob().model_config, parameters: {n_estimators: 2000}}})];
    assert.throws(() => get("campaignBody")(), /2000 деревьев/);
  } else if (which === "source_filter") {
    const state = studySetup(); state.campaignSourceJobs = [sourceJob(), sourceJob({id: otherId, source: "imported"}), sourceJob({id: "e".repeat(32), status: "failed"}), sourceJob({id: "f".repeat(32), task_id: "reg"}), sourceJob({id: "1".repeat(32), model_config: {model: "FixedBlend"}})];
    assert.deepEqual(plain(get("lightgbmStudySources")("cls").map((job) => job.id)), [id]);
    state.campaignDraft.source_job_id = otherId;
    assert.throws(() => get("campaignBody")(), /недоступен/);
  } else if (which === "stale_study_preview") {
    const state = studySetup(); let finish;
    responder = () => new Promise((resolve) => {finish = resolve;});
    const request = get("previewCampaign")(); await new Promise((resolve) => setImmediate(resolve));
    get("updateLightgbmStudy")(id, 45);
    finish({ready: true, steps: [], source_seal: {valid: true}, runtime_seal: {valid: true}, plan_sha256: "d".repeat(64)});
    await request;
    assert.equal(state.campaignPreview, null); assert.equal(state.campaignPreviewBody, null);
    assert.equal(state.campaignDraft.minutes_per_step, 45);
    assert.match(get("campaignLaunchBlock")(), /проверьте план/);
  } else if (which === "reviewed_recipes") {
    const state = studySetup();
    const steps = get("lightgbmStudySteps")(["cls"], id, 60).map((step) => ({...step, model_config: {...sourceJob().model_config, parameters: step.model_params}, worker_timeout_seconds: 3600}));
    state.campaignPreview = {ready: true, steps, experiment: {kind: "lightgbm_tree_count", control_description: "Замороженный контроль"}};
    const html = get("campaignPreviewHTML")();
    assert.match(html, /Что именно будет обучаться/); assert.match(html, /750/); assert.match(html, /1000/);
    assert.match(html, /class_weight/); assert.match(html, /balanced/); assert.match(html, /Контроль/);
    assert.match(html, /метрика получена до него/);
    responder = () => state.campaignPreview; await get("previewCampaign")();
    get("selectCampaignPreset")("custom");
    const body = plain(get("campaignBody")());
    assert.equal(body.steps[0].minutes, 60); assert.equal(body.steps[0].feature_config.signed_log, true);
    assert.equal(body.steps[0].model_config.parameters.class_weight, "balanced");
    assert.equal("model_params" in body.steps[0], false, "A canonical recipe cannot be submitted alongside conflicting shorthand parameters");
  } else if (which === "control_comparison") {
    studySetup(); const group = {task_id: "cls", comparison_key: "c".repeat(64), dataset_fingerprint: "d".repeat(64), metric_name: "Macro F1", higher_is_better: true,
      comparison_available: true, best_job_id: otherId, control_job_id: id, gain: .01,
      jobs: [{job_id: id, action: "lightgbm", minutes: 30, value: .62, elapsed_seconds: 50, is_control: true, label: "Контроль · 500", improvement: 0},
        {job_id: otherId, action: "lightgbm", minutes: 30, value: .63, elapsed_seconds: 90, label: "Вариант · 750", improvement: .01}]};
    let html = get("campaignComparisonsHTML")([group]);
    assert.match(html, /Улучшение к контролю/); assert.match(html, /Фактически/); assert.match(html, /\+0[,.]01/);
    html = get("campaignComparisonsHTML")([{...group, comparison_available: false, comparison_reason: "control_not_admitted", control_job_id: null}]);
    assert.match(html, /Контроль не прошёл/); assert.doesNotMatch(html, /\+0[,.]01|★/);
    assert.doesNotMatch(get("campaignComparisonsHTML")([{...group, comparison_key: null}]), /\+0[,.]01|★/);
  } else if (which === "artifact_downloads") {
    studySetup(); const job = sourceJob({action: "lightgbm", model_artifact: {available: true, format: "joblib", sha256: "e".repeat(64), size_bytes: 2048}});
    const html = get("trainedModelView")(job);
    assert.match(html, new RegExp(`/api/jobs/${id}/model\\?task_id=cls`));
    assert.match(html, /model-manifest\?task_id=cls/); assert.match(html, /recipe\?task_id=cls/);
    assert.match(html, /Финальное обучение на полном train/); assert.match(html, /код конкурсного агента отдельно/);
    assert.equal(get("trainedModelView")({...job, source: "imported"}), "");
    assert.equal(get("trainedModelView")({...job, model_artifact: {...job.model_artifact, sha256: "bad"}}), "");
    assert.equal(get("trainedModelView")({...job, status: "failed"}), "");
    const absent = get("trainedModelView")(sourceJob({model_artifact: {available: false, reason: "AutoML не сохраняет файл модели <script>"}}));
    assert.match(absent, /не сохраняет файл/); assert.doesNotMatch(absent, /href=|<script>/);
  } else if (which === "gpu_controls") {
    const state = studySetup(); let html = get("trainingDeviceControlsHTML")();
    assert.match(html, /value="gpu"[^>]*disabled/); assert.match(html, /CPU доступен сразу/);
    responder = (url) => {assert.equal(url, "/api/training-devices"); return {gpu: {available: true, device_name: "RTX 3080", training_verified: false}};};
    await get("refreshTrainingDevices")(); html = get("trainingDeviceControlsHTML")();
    assert.doesNotMatch(html, /value="gpu"[^>]*disabled/); assert.match(html, /Проверка доступности не запускает обучение/);
    assert.equal(calls.length, 1); assert.equal(calls[0].options?.method, undefined);
    const values = {task_id: "cls", action: "catboost", author: "", hypothesis: "", minutes: 30, training_device: "gpu"};
    assert.equal(get("launchPayload")(values).training_device, "gpu");
    assert.equal("training_device" in get("launchPayload")({...values, action: "lightgbm"}), false);
    state.trainingDevices = {gpu: {available: false, reason: '<img src=x onerror="evil()">'}};
    assert.doesNotMatch(get("trainingDeviceControlsHTML")(), /<img/);
  } else if (which === "gpu_evidence") {
    studySetup(); const job = sourceJob({training_device: "gpu"});
    assert.doesNotMatch(get("trainingDeviceEvidenceView")(job), /Обучено на GPU/);
    const evidence = {fit_completed: true, backend_verified: true, backend: "GPU", device_name: "RTX 3080"};
    assert.match(get("trainingDeviceEvidenceView")({...job, training_device_evidence: evidence}), /Обучено на GPU/);
    assert.doesNotMatch(get("trainingDeviceEvidenceView")({...job, training_device_evidence: {...evidence, backend_verified: false}}), /Обучено на GPU/);
  } else throw new Error("Unknown training UI check " + which);
}
main().catch((error) => {console.error(error.stack || error); process.exitCode = 1;});
'''


@unittest.skipUnless(NODE, "Training UI checks require Node.js")
class TrainingFrontendTests(unittest.TestCase):
    def test_training_studies_artifacts_and_device_boundaries(self):
        for case in ("study_payload", "inherit_source", "source_filter", "stale_study_preview", "reviewed_recipes",
                     "control_comparison", "artifact_downloads", "gpu_controls", "gpu_evidence"):
            with self.subTest(case=case):
                result = subprocess.run([NODE, "-e", SCRIPT, str(ROOT / "app/static/app.js"), case],
                                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
