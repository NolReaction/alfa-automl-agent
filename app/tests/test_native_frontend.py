"""Exercise native-series UI behavior with a browser-shell stub, without training."""
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
SCRIPT = r'''
"use strict";
const fs = require("node:fs"), vm = require("node:vm"), assert = require("node:assert/strict");
const noop = () => {}, nodes = new Map(), launchButtons = [];
function element(selector) {
  if (!nodes.has(selector)) nodes.set(selector, {
    innerHTML: "", textContent: "", value: "", disabled: false, hidden: false,
    dataset: {}, elements: {}, options: [], children: [], isConnected: true,
    scrollTop: 0, clientHeight: 100, scrollHeight: 100,
    classList: {add: noop, remove: noop, toggle: noop},
    addEventListener: noop, setAttribute: noop, removeAttribute: noop,
    append: noop, insertAdjacentHTML: noop, close: noop, showModal: noop,
    querySelector: (name) => element(name), querySelectorAll: () => [],
  });
  return nodes.get(selector);
}
const calls = [], tasks = [
  {id: "reg", name: "Regression", available: true, task_type: "regression"},
  {id: "cls", name: "Classification", available: true, task_type: "classification"},
  {id: "missing", name: "Unavailable", available: false, task_type: "regression"},
];
let responder = (url) => url === "/api/tasks" ? {tasks: []} :
  url === "/api/session" ? {token: "test-token", api: {ready: true}} : {queues: [], active_queue: null};
const context = vm.createContext({
  console, document: {hidden: true, activeElement: null,
    querySelector: (query) => element(query),
    querySelectorAll: (query) => query === '[data-launch]' ? launchButtons : [],
    addEventListener: noop, createElement: (tag) => element(tag), body: element("body")},
  window: {addEventListener: noop}, location: {hash: ""}, history: {replaceState: noop}, navigator: {},
  localStorage: {getItem: () => null, setItem: noop}, URLSearchParams, structuredClone,
  setInterval: () => 0, setTimeout: () => 0, clearTimeout: noop,
  fetch: async (url, options) => {calls.push({url, options}); return {ok: true, json: async () => responder(url)};},
});
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context, {filename: process.argv[1], timeout: 5000});
const get = (name) => vm.runInContext(name, context), plain = (value) => JSON.parse(JSON.stringify(value));
const id = "a".repeat(32), otherId = "b".repeat(32);
function setup() {
  const state = get("state");
  state.tasks = structuredClone(tasks); state.task = null; state.view = "native"; state.generation = 20;
  state.workflow = null; state.session = {token: "test-token", api: {ready: true}};
  state.nativePreflight = {supported: true, ready: true, errors: [], release_sha: "c".repeat(40), threads: 8};
  state.nativeQueues = []; state.nativeActiveQueue = null; state.nativeSelectedQueue = null;
  state.nativeStartBusy = false; state.nativePollBusy = false;
  state.nativeRegressionTask = "reg"; state.nativeClassificationTask = "cls";
  state.nativeBudgetSeconds = 10500; state.nativeLogKind = "coordinator";
  state.nativeLogProfile = "regression_off"; calls.length = 0;
  return state;
}
async function main() {
  await new Promise((resolve) => setImmediate(resolve));
  const which = process.argv[2];
  if (which === "payload") {
    setup();
    assert.deepEqual(plain(get("nativePayload")("reg", "cls", 10500)), {
      regression_task_id: "reg", classification_task_id: "cls", time_budget_seconds: 10500,
    });
    for (const seconds of [300, 10501, 3600.5, NaN, "10500"])
      assert.throws(() => get("nativePayload")("reg", "cls", seconds));
    for (const pair of [["missing", "cls"], ["cls", "reg"], ["reg", "absent"]])
      assert.throws(() => get("nativePayload")(...pair, 3600));
  } else if (which === "gates") {
    const state = setup();
    assert.equal(get("nativeLaunchBlock")(), null);
    state.nativePreflight = null; assert.match(get("nativeLaunchBlock")(), /готовность/);
    state.nativePreflight = {ready: false}; assert.match(get("nativeLaunchBlock")(), /причины/);
    state.nativePreflight = {ready: true};
    state.nativeActiveQueue = {id, status: "running"}; assert.match(get("nativeLaunchBlock")(), /текущей серии/);
    const before = state.launchGeneration;
    await get("showLaunch")("baseline");
    assert.equal(state.launchGeneration, before, "An ordinary launch cannot open while the series owns CPU");
    state.nativeActiveQueue = null; state.session.active_job = {status: "running"};
    assert.match(get("nativeLaunchBlock")(), /другой запуск/);
  } else if (which === "admission_resources") {
    setup();
    const pending = get("nativeProfileCard")({profile: "regression_on", status: "completed",
      qualification: "requires_review", continuation_allowed: true,
      resources: {available: false, peak_rss_bytes: 12 * 1024 ** 3}, checks: {owned_cleanup_complete: true}});
    assert.match(pending, /Обязательные проверки пройдены/);
    assert.match(pending, /требует разбора/);
    assert.match(pending, /Измерения ресурсов пока не подтверждены/);
    assert.doesNotMatch(pending, /12 ГиБ|Измеренный пик RAM:/);
    const failed = get("nativeProfileCard")({profile: "regression_off", status: "failed",
      continuation_allowed: false, resources: {available: true, peak_rss_bytes: 2 * 1024 ** 3}});
    assert.match(failed, /Переход к следующему заданию запрещён/);
    assert.match(failed, /Измеренный пик RAM: 2 ГиБ/);
  } else if (which === "escaping") {
    setup();
    const html = get("nativeDetailHTML")({id, status: "failed", created_at: "2026-10-23T01:00:00Z",
      time_budget_seconds: 10500, release_sha: '<img src=x onerror="evil()">',
      error: '<script>evil()</script>', output_dir: '<svg/onload=evil()>',
      profiles: [{profile: "classification", status: "failed", error: '<img src=x onerror="evil()">',
        stage: '<svg/onload=evil()>', metric: {name: '<script>evil()</script>', value: .6},
        checks: {'<script>evil()</script>': false}}]});
    assert.doesNotMatch(html, /<script>|<img|<svg\/onload/);
    assert.match(html, /&lt;script&gt;/);
    assert.throws(() => get("nativePath")("../../secret"));
    assert.throws(() => get("nativePath")('" onclick="evil()'));
  } else if (which === "comparison") {
    setup();
    const pair = {status: "comparable", off: {name: "RMSE", value: 10},
      on: {name: "RMSE", value: 9}, on_minus_off: -1, rescue_exercised: false};
    const html = get("nativeComparison")(pair);
    assert.match(html, /ON − OFF/); assert.match(html, /Rescue был пропущен/);
    assert.match(html, /Одна пара.*не доказывает общий прирост/);
    assert.doesNotMatch(get("nativeComparison")({...pair, status: "incompatible"}), /RMSE · ON − OFF/);
    assert.match(get("nativeComparison")(null), /после двух успешно проверенных/);
  } else if (which === "stopped_queue") {
    setup();
    const html = get("nativeDetailHTML")({id, status: "cancelled", time_budget_seconds: 10500,
      profiles: [{profile: "regression_off", status: "interrupted"},
        {profile: "regression_on", status: "pending"}, {profile: "classification", status: "pending"}]});
    assert.equal((html.match(/Не запускалось/g) || []).length, 2);
    assert.doesNotMatch(html, /Ожидает очереди/);
    assert.doesNotMatch(html, /data-action="native-stop"/);
  } else if (which === "cleanup_pending") {
    const state = setup(), queue = {id, status: "cleanup_pending", time_budget_seconds: 10500,
      profiles: [{profile: "regression_off", status: "interrupted"}]};
    get("setNativeActive")(queue);
    assert.equal(get("nativeRunning")(queue), true);
    assert.equal(state.nativeActiveQueue.id, id);
    assert.match(get("nativeLaunchBlock")(), /текущей серии/);
    const html = get("nativeDetailHTML")(queue);
    assert.match(html, /Остановка требует проверки/);
    assert.match(html, /Повторить остановку/);
    assert.match(html, /data-action="native-stop"[^>]*>/);
    assert.doesNotMatch(html, /data-action="native-stop"[^>]*disabled/);
    assert.equal((html.match(/Не запускалось/g) || []).length, 2);
    const button = element("cleanup-launch"); launchButtons.push(button);
    get("applyNativeLaneControls")(); assert.equal(button.disabled, true);
  } else if (which === "lane_buttons") {
    const state = setup();
    const enabled = element("enabled"), unavailable = element("unavailable");
    enabled.disabled = false; unavailable.disabled = true; launchButtons.push(enabled, unavailable);
    state.nativeActiveQueue = {id, status: "running"}; get("applyNativeLaneControls")();
    assert.equal(enabled.disabled, true); assert.equal(unavailable.disabled, true);
    state.nativeActiveQueue = null; get("applyNativeLaneControls")();
    assert.equal(enabled.disabled, false); assert.equal(unavailable.disabled, true,
      "Ending a series must not enable a task button that was unavailable before it");
  } else if (which === "poll_scope") {
    const state = setup(); context.document.hidden = false;
    responder = () => ({queues: [], active_queue: null});
    state.view = "data"; await get("pollNativeQueues")(); assert.equal(calls.length, 0);
    state.view = "native"; state.nativePollBusy = true;
    await get("pollNativeQueues")(); assert.equal(calls.length, 0);
    state.nativePollBusy = false; await get("pollNativeQueues")();
    assert.deepEqual(calls.map((call) => call.url).sort(), ["/api/native-queues", "/api/session"]);
    assert.equal(calls.some((call) => call.url.endsWith("/preflight")), false,
      "Polling must not rerun environment or dataset preflight");
  } else if (which === "ordinary_completion") {
    const state = setup(); context.document.hidden = false;
    state.workflow = {active_job: {id: "ordinary", status: "running"}};
    state.session.active_job = {id: "ordinary", status: "running"};
    assert.match(get("nativeLaunchBlock")(), /другой запуск/);
    responder = (url) => url === "/api/session"
      ? {token: "test-token", active_job: null, api: {ready: true}}
      : {queues: [], active_queue: null};
    await get("pollNativeQueues")();
    assert.equal(state.workflow.active_job.status, "running", "The stale task workflow remains independent");
    assert.equal(state.session.active_job, null);
    assert.equal(get("nativeLaunchBlock")(), null,
      "A completed ordinary job must not leave the native launch permanently blocked by its old task workflow");
    assert.equal(element("#native-start").disabled, false);
  } else if (which === "stale_detail") {
    const state = setup(); let finish;
    state.nativeSelectedQueue = id; state.nativeDetail = {id: otherId};
    responder = () => new Promise((resolve) => {finish = resolve;});
    const pending = get("loadNativeDetail")(id, 20);
    await new Promise((resolve) => setImmediate(resolve));
    state.nativeSelectedQueue = otherId; state.generation++;
    finish({queue: {id, status: "completed", profiles: []}}); await pending;
    assert.equal(state.nativeDetail.id, otherId, "A stale detail response must not overwrite another selected series");
  } else if (which === "no_selected_task") {
    const state = setup(); state.tasks = []; state.task = null;
    responder = (url) => url === "/api/tasks" ? {tasks: []} : {queues: [], active_queue: null};
    await get("loadNativePage")(state.generation);
    assert.match(element("#main").innerHTML, /Ночные тесты/);
    assert.match(element("#main").innerHTML, /Нет готовой задачи/);
    assert.equal(calls.some((call) => /\/workflow$/.test(call.url)), false);
    assert.equal(element("#task-select").disabled, true);
  } else throw new Error("Unknown frontend test " + which);
}
main().catch((error) => {console.error(error.stack || error); process.exitCode = 1;});
'''


@unittest.skipUnless(NODE, "Native UI checks require Node.js")
class NativeFrontendTests(unittest.TestCase):
    def test_native_ui_decisions_and_request_boundaries(self):
        for case in ("payload", "gates", "admission_resources", "escaping", "comparison", "stopped_queue", "cleanup_pending",
                     "lane_buttons", "poll_scope", "ordinary_completion", "stale_detail", "no_selected_task"):
            with self.subTest(case=case):
                result = subprocess.run([NODE, "-e", SCRIPT, str(ROOT / "app/static/app.js"), case],
                                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_navigation_is_available_outside_current_task(self):
        html = (ROOT / "app/static/index.html").read_text(encoding="utf-8")
        self.assertIn('data-view="native"', html)
        self.assertIn("Ночные тесты", html)


if __name__ == "__main__":
    unittest.main()
