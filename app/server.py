"""Run the local Alfa Lab workspace: python app/server.py [--port 8765]."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.profiling import DataCatalog
from app.runs import RunReader

VERSION = "0.1.0"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, data_root: Path, port: int = 8765):
        self.catalog = DataCatalog(Path(data_root))
        self.readers: dict[str, RunReader] = {}
        self.reader_lock = threading.Lock()
        # Deliberately local-only; no host option or external bind address.
        super().__init__(("127.0.0.1", port), WorkspaceHandler)

    def reader(self, task_id: str) -> RunReader:
        task_dir = self.catalog.resolve(task_id)
        with self.reader_lock:
            key = str(task_dir)
            if key not in self.readers:
                if len(self.readers) >= 16:
                    self.readers.pop(next(iter(self.readers)))
                self.readers[key] = RunReader(task_dir)
            return self.readers[key]


class WorkspaceHandler(BaseHTTPRequestHandler):
    server: WorkspaceServer
    server_version = "AlfaLab/" + VERSION
    sys_version = ""

    def log_message(self, format: str, *args) -> None:
        # Request strings are untrusted and can contain terminal control codes.
        message = (format % args).encode("unicode_escape").decode("ascii")
        logging.info("%s", message)

    def _headers(self, status: int, content_type: str, length: int, *, download=False):
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
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="submission.csv"')
        self.end_headers()

    def _bytes(self, body: bytes, content_type: str, status=200):
        self._headers(status, content_type, len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._bytes(body, "application/json; charset=utf-8", status)

    def _local_request(self) -> bool:
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "").lower()
        if host not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() != f"http://{host}":
            return False
        return self.headers.get("Sec-Fetch-Site", "none") in {"none", "same-origin"}

    def do_GET(self):
        try:
            if not self._local_request():
                self._json({"error": "Доступ разрешён только со страницы локального приложения."}, 403)
                return
            if len(self.path) > 4096:
                self._json({"error": "Слишком длинный запрос."}, 414)
                return
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc:
                raise ValueError("Неверный адрес запроса.")
            path = unquote(parsed.path, errors="strict")
            if "\x00" in path or "\\" in path or any(p in {".", ".."} for p in path.split("/")):
                raise ValueError("Недопустимый путь.")
            query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=8)
            if path in STATIC:
                name, content_type = STATIC[path]
                self._bytes((APP_DIR / "static" / name).read_bytes(), content_type)
                return
            if path == "/favicon.ico":
                self._bytes(b"", "image/x-icon", 204)
                return
            if path == "/api/tasks":
                self._json({"tasks": self.server.catalog.discover(), "version": VERSION})
                return
            parts = path.strip("/").split("/")
            if len(parts) < 4 or parts[:2] != ["api", "tasks"] or not SAFE_ID.fullmatch(parts[2]):
                raise FileNotFoundError("Страница не найдена.")
            task_id, action = parts[2], parts[3]
            # Resolve every request against the current discovered catalog.
            self.server.catalog.resolve(task_id)
            if len(parts) == 4:
                if action == "profile":
                    self._json(self.server.catalog.profile(task_id))
                elif action == "column":
                    self._json(self.server.catalog.column(task_id, self._one(query, "name")))
                elif action == "description":
                    self._json({"text": self.server.catalog.task_text(task_id)})
                elif action == "runs":
                    self._json({"runs": self.server.reader(task_id).list_runs()})
                elif action == "result":
                    self._json(self.server.reader(task_id).result())
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
        except (BrokenPipeError, ConnectionResetError):
            return
        except FileNotFoundError as error:
            self._json({"error": str(error)}, 404)
        except PermissionError:
            self._json({"error": "Этот файл недоступен приложению."}, 403)
        except (ValueError, KeyError, UnicodeError) as error:
            self._json({"error": str(error)}, 400)
        except OSError:
            self._json({"error": "Не удалось прочитать файл. Проверьте, что он существует и доступен."}, 409)
        except Exception:
            logging.exception("Workspace request failed")
            self._json({"error": "Не удалось обработать данные. Подробности доступны в терминале приложения."}, 500)

    def _download(self, path: Path):
        # RunReader returns only the exact, checked task-local submission path.
        with path.open("rb") as stream:
            import os
            size = os.fstat(stream.fileno()).st_size
            if size > 128 * 1024 * 1024:
                raise ValueError("Файл слишком большой для скачивания через панель.")
            self._headers(200, "text/csv; charset=utf-8", size, download=True)
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

    def do_POST(self):
        self._json({"error": "Панель работает в режиме чтения."}, 405)

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST
    do_OPTIONS = do_POST


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Directory containing task folders")
    parser.add_argument("--open", action="store_true", help="Open the local page in your browser")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        with WorkspaceServer(args.data_dir.expanduser().absolute(), args.port) as server:
            url = f"http://127.0.0.1:{server.server_port}/"
            print(f"Alfa Lab: {url}", flush=True)
            print(f"Data: {args.data_dir}", flush=True)
            print("Local read-only workspace. Press Ctrl+C to stop.", flush=True)
            if args.open:
                webbrowser.open(url)
            server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nAlfa Lab stopped.")
    except OSError as error:
        print(f"Cannot start Alfa Lab: {error}. Try --port 8766.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
