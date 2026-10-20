"""Run Alfa Lab locally: python app/server.py [--port 8765]."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import secrets
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.jobs import JobManager, api_settings
from app.model_config import model_options
from app.exports import delivery_archive, result_archive, report_bytes_from_archive, MAX_IMPORT_BYTES
from app.profiling import DataCatalog
from app.runs import RunReader
from app.sharing import ReportStore

VERSION = "0.5.1"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MAX_BODY = 1024 * 1024


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, data_root: Path, port: int = 8765, state_root: Path | None = None):
        self.catalog = DataCatalog(Path(data_root))
        self.readers: dict[str, RunReader] = {}
        self.reader_lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.state_root = Path(state_root) if state_root is not None else APP_DIR / ".workspace"
        self.jobs = JobManager(Path(data_root), self.state_root, self.catalog)
        self.reports = ReportStore(self.state_root)
        try:
            super().__init__(("127.0.0.1", port), WorkspaceHandler)
        except Exception:
            self.jobs.close()
            raise

    def server_close(self):
        try:
            self.jobs.close()
        finally:
            super().server_close()

    def reader(self, task_id: str) -> RunReader:
        task_dir = self.catalog.resolve(task_id)
        with self.reader_lock:
            key = str(task_dir)
            if key not in self.readers:
                if len(self.readers) >= 16:
                    self.readers.pop(next(iter(self.readers)))
                self.readers[key] = RunReader(task_dir)
            return self.readers[key]

    def experiments(self, task_id):
        return sorted(self.jobs.list(task_id) + self.reports.list(task_id),
                      key=lambda item: item.get("started_at") or "", reverse=True)[:400]

    def workflow(self, task_id):
        task = next((item for item in self.catalog.discover() if item["id"] == task_id), None)
        if task is None:
            raise FileNotFoundError("Задача не найдена.")
        result = self.reader(task_id).result()
        baseline = result["baseline"]
        jobs = self.jobs.list(task_id)
        try:
            fingerprint = self.catalog.fingerprint(task_id)
        except (OSError, ValueError):
            fingerprint = None
        return {
            "task": task, "dataset_fingerprint": fingerprint,
            "baseline": ({"exists": True, "metric": baseline.get("metric"),
                          "model": baseline.get("model", {}).get("name")} if baseline else None),
            "submission": {name: result["submission"].get(name) for name in ("exists", "valid")},
            "api": api_settings(), "active_job": self.jobs.active_job(),
            "latest_job": jobs[0] if jobs else None,
            "experiments_count": len(jobs) + len(self.reports.list(task_id)),
            "published_job_id": next((item["id"] for item in jobs if item.get("published")), None),
        }


class WorkspaceHandler(BaseHTTPRequestHandler):
    server: WorkspaceServer
    server_version = "AlfaLab/" + VERSION
    sys_version = ""

    def log_message(self, format: str, *args) -> None:
        logging.info("%s", (format % args).encode("unicode_escape").decode("ascii"))

    def _headers(self, status, content_type, length, *, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        ))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()

    def _bytes(self, body, content_type, status=200, *, filename=None):
        self._headers(status, content_type, len(body), filename=filename)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value, status=200, *, filename=None):
        # ReportStore bounds this compact representation. Extra whitespace can
        # otherwise make our own near-limit download too large to import.
        body = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
        self._bytes(body, "application/json; charset=utf-8", status, filename=filename)

    def _local_request(self):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "").lower()
        if host not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() != f"http://{host}":
            return False
        return self.headers.get("Sec-Fetch-Site", "none") in {"none", "same-origin"}

    def _address(self):
        if len(self.path) > 4096:
            raise ValueError("Слишком длинный запрос.")
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc:
            raise ValueError("Неверный адрес запроса.")
        path = unquote(parsed.path, errors="strict")
        if "\x00" in path or "\\" in path or any(p in {".", ".."} for p in path.split("/")):
            raise ValueError("Недопустимый путь.")
        query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=8)
        return path, path.strip("/").split("/"), query

    def _error(self, error):
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        if isinstance(error, FileNotFoundError):
            self._json({"error": str(error)}, 404)
        elif isinstance(error, PermissionError):
            self._json({"error": "Доступ к действию или файлу запрещён."}, 403)
        elif isinstance(error, (ValueError, KeyError, UnicodeError, TypeError, RecursionError)):
            self._json({"error": str(error) or "Некорректный запрос."}, 400)
        elif isinstance(error, (RuntimeError, FileExistsError)):
            self._json({"error": str(error)}, 409)
        elif isinstance(error, OSError):
            self._json({"error": "Не удалось прочитать или сохранить файл. Проверьте доступ к папке."}, 409)
        else:
            logging.exception("Workspace request failed")
            self._json({"error": "Не удалось обработать запрос. Подробности доступны в терминале приложения."}, 500)

    def do_GET(self):
        try:
            if not self._local_request():
                raise PermissionError()
            path, parts, query = self._address()
            if path in STATIC:
                name, content_type = STATIC[path]
                self._bytes((APP_DIR / "static" / name).read_bytes(), content_type)
                return
            if path == "/favicon.ico":
                self._bytes(b"", "image/x-icon", 204)
                return
            if path == "/api/session":
                self._json({"token": self.server.token, "api": api_settings(),
                            "active_job": self.server.jobs.active_job(), "version": VERSION})
                return
            if path == "/api/tasks":
                self._json({"tasks": self.server.catalog.discover(), "version": VERSION})
                return
            if parts[:2] == ["api", "jobs"]:
                task_id = self._query_task(query)
                if len(parts) == 2:
                    self._json({"jobs": self.server.jobs.list(task_id)})
                    return
                if (len(parts) in {4, 5, 6} and SAFE_ID.fullmatch(parts[2])
                        and parts[3] == "agent-runs"):
                    reader = self.server.jobs.agent_reader(task_id, parts[2])
                    if len(parts) == 4:
                        value = {"runs": reader.list_runs()}
                    elif SAFE_ID.fullmatch(parts[4]):
                        if len(parts) == 5:
                            value = reader.detail(parts[4])
                        elif parts[5] == "file":
                            value = reader.file(parts[4], self._one(query, "name"))
                        else:
                            raise FileNotFoundError("Файл запуска не найден.")
                    else:
                        raise FileNotFoundError("Запуск агента не найден.")
                    self._json(self.server.jobs.redact(task_id, parts[2], value))
                    return
                if len(parts) in {3, 4} and SAFE_ID.fullmatch(parts[2]):
                    if len(parts) == 3:
                        self._json(self.server.jobs.detail(task_id, parts[2]))
                        return
                    if parts[3] == "submission":
                        self._download(self.server.jobs.artifact(task_id, parts[2]))
                        return
                    if parts[3] == "bundle":
                        body = result_archive(task_id, parts[2], jobs=self.server.jobs,
                                              catalog=self.server.catalog, reports=self.server.reports)
                        self._bytes(body, "application/zip", filename=f"alfa-lab-result-{parts[2][:8]}.zip")
                        return
                raise FileNotFoundError("Запуск не найден.")
            if len(parts) < 4 or parts[:2] != ["api", "tasks"] or not SAFE_ID.fullmatch(parts[2]):
                raise FileNotFoundError("Страница не найдена.")
            task_id, action = parts[2], parts[3]
            self.server.catalog.resolve(task_id)
            if len(parts) == 4:
                if action == "profile":
                    profile = self.server.catalog.profile(task_id)
                    self._json({**profile, "model_options": model_options(profile["task_type"])})
                elif action == "column":
                    self._json(self.server.catalog.column(task_id, self._one(query, "name")))
                elif action == "description":
                    self._json({"text": self.server.catalog.task_text(task_id)})
                elif action == "runs":
                    self._json({"runs": self.server.reader(task_id).list_runs()})
                elif action == "result":
                    self._json(self.server.reader(task_id).result())
                elif action == "workflow":
                    self._json(self.server.workflow(task_id))
                elif action == "experiments":
                    self._json({"experiments": self.server.experiments(task_id),
                                "dataset_fingerprint": self.server.catalog.fingerprint(task_id),
                                "active_job": self.server.jobs.active_job()})
                elif action == "export":
                    envelope = self.server.reports.export(task_id, self.server.catalog.fingerprint(task_id),
                                                          self.server.experiments(task_id))
                    self._json(envelope, filename="alfa-lab-reports.json")
                elif action == "submission":
                    self._download(self.server.reader(task_id).submission_path())
                else:
                    raise FileNotFoundError("Страница не найдена.")
                return
            if action == "runs" and SAFE_ID.fullmatch(parts[4]):
                reader, run_id = self.server.reader(task_id), parts[4]
                if len(parts) == 5:
                    self._json(reader.detail(run_id))
                    return
                if len(parts) == 6 and parts[5] == "file":
                    self._json(reader.file(run_id, self._one(query, "name")))
                    return
            raise FileNotFoundError("Страница не найдена.")
        except Exception as error:
            self._error(error)

    def _body(self, *, allow_zip=False):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Передача частями не поддерживается.")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise ValueError("Не указан размер запроса.")
        length = int(lengths[0])
        is_zip = allow_zip and self.headers.get_content_type() == "application/zip"
        limit = MAX_IMPORT_BYTES if is_zip else MAX_BODY
        if length > limit:
            raise ValueError("ZIP отчёта превышает 20 МБ." if is_zip else "Файл отчёта превышает 1 МБ.")
        if not is_zip and self.headers.get_content_type() != "application/json":
            raise ValueError("Ожидается JSON или ZIP отчёта." if allow_zip else "Ожидается JSON-запрос.")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("Запрос получен не полностью.")
        if is_zip:
            raw = report_bytes_from_archive(raw)
        def reject(value):
            raise ValueError("Недопустимое числовое значение в JSON.")
        value = json.loads(raw.decode("utf-8"), parse_constant=reject)
        if not isinstance(value, dict):
            raise ValueError("Ожидается JSON-объект.")
        return value

    def do_POST(self):
        try:
            if not self._local_request():
                raise PermissionError()
            path, parts, query = self._address()
            is_job = path == "/api/jobs"
            is_delivery = path == "/api/delivery"
            is_job_action = (len(parts) == 4 and parts[:2] == ["api", "jobs"]
                             and SAFE_ID.fullmatch(parts[2]) and parts[3] in {"stop", "publish"})
            is_import = (len(parts) == 4 and parts[:2] == ["api", "tasks"]
                         and SAFE_ID.fullmatch(parts[2]) and parts[3] == "import")
            if not (is_job or is_job_action or is_import or is_delivery):
                self._json({"error": "Это действие не поддерживается."}, 405)
                return
            token = self.headers.get("X-Alfa-Token", "")
            if not hmac.compare_digest(token, self.server.token):
                raise PermissionError()
            body = self._body(allow_zip=is_import)
            if is_delivery:
                if body:
                    raise ValueError("Сборка решения не принимает дополнительные параметры.")
                self._bytes(delivery_archive(), "application/zip", filename="solution.zip")
            elif is_job:
                if set(body) - {"task_id", "action", "author", "hypothesis", "minutes", "validation", "model_params"}:
                    raise ValueError("Неизвестные параметры запуска.")
                if "validation" in body and body.get("action") not in ("baseline", "extra_trees", "catboost", "lightgbm", "linear", "automl"):
                    raise ValueError("Выбор валидации доступен только для локальных моделей и автоподбора.")
                if "model_params" in body and body.get("action") not in ("baseline", "extra_trees", "catboost", "lightgbm", "linear"):
                    raise ValueError("Настройка модели доступна только для baseline и ExtraTrees.")
                task_id = self._task_value(body.get("task_id"), allow_none=True)
                job = self.server.jobs.start(task_id, body.get("action"),
                                             author=body.get("author", ""),
                                             hypothesis=body.get("hypothesis", ""),
                                             minutes=body.get("minutes", 10 if body.get("action") == "automl" else 20),
                                             validation=body.get("validation"),
                                             model_params=body.get("model_params"))
                self._json({"job": job}, 202)
            elif is_job_action:
                if set(body) != {"task_id"}:
                    raise ValueError("Укажите задачу выбранного запуска.")
                task_id = self._task_value(body.get("task_id"), allow_none=True)
                action = self.server.jobs.stop if parts[3] == "stop" else self.server.jobs.publish
                self._json({"job": action(task_id, parts[2])})
            else:
                task_id = self._task_value(parts[2])
                self._json(self.server.reports.import_reports(task_id,
                           self.server.catalog.fingerprint(task_id), body))
        except Exception as error:
            self._error(error)

    def _task_value(self, value, allow_none=False):
        if value is None and allow_none:
            return None
        if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
            raise ValueError("Неверный идентификатор задачи.")
        self.server.catalog.resolve(value)
        return value

    def _query_task(self, query):
        if "task_id" not in query:
            return None
        return self._task_value(self._one(query, "task_id"))

    def _download(self, path):
        with path.open("rb") as stream:
            import os
            size = os.fstat(stream.fileno()).st_size
            if size > 128 * 1024 * 1024:
                raise ValueError("Файл слишком большой для скачивания через панель.")
            self._headers(200, "text/csv; charset=utf-8", size, filename="submission.csv")
            if self.command != "HEAD":
                remaining = size
                while remaining:
                    block = stream.read(min(64 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    @staticmethod
    def _one(query, name):
        values = query.get(name, [])
        if len(values) != 1 or not values[0] or len(values[0]) > 512:
            raise ValueError("Не задано корректное имя файла или столбца.")
        return values[0]

    def do_HEAD(self):
        self.do_GET()

    def _unsupported(self):
        self._json({"error": "Это действие не поддерживается."}, 405)

    do_PUT = _unsupported
    do_PATCH = _unsupported
    do_DELETE = _unsupported
    do_OPTIONS = _unsupported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--state-dir", type=Path, default=APP_DIR / ".workspace")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        with WorkspaceServer(args.data_dir.expanduser().absolute(), args.port,
                             args.state_dir.expanduser().absolute()) as server:
            url = f"http://127.0.0.1:{server.server_port}/"
            print(f"Alfa Lab: {url}\nData: {args.data_dir}\nPress Ctrl+C to stop.", flush=True)
            if args.open:
                webbrowser.open(url)
            server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nAlfa Lab stopped.")
    except ValueError as error:
        print(f"Cannot start Alfa Lab: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Cannot start Alfa Lab: {error}. Try --port 8766.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
