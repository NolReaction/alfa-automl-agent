"""Desktop control window for the local laboratory (no console required)."""
from __future__ import annotations

import errno
from pathlib import Path
import sys
import threading
import time
import webbrowser


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def project_paths(app_dir: Path = APP_DIR) -> tuple[Path, Path]:
    """Use this checkout's data and persistent history regardless of the cwd."""
    return app_dir.parent / "data", app_dir / ".workspace"


def open_server(server_type, data_root: Path, state_root: Path):
    """A busy TCP port may belong to another app; never attach to it."""
    try:
        return server_type(data_root, 8765, state_root)
    except OSError as error:
        if error.errno != errno.EADDRINUSE and getattr(error, "winerror", None) != 10048:
            raise
        return server_type(data_root, 0, state_root)


def close_server(server, thread) -> None:
    if server is None:
        return
    try:
        if thread is not None and thread.is_alive():
            server.shutdown()
    finally:
        server.server_close()
        # JobManager.close requests cancellation but its first wait is bounded.
        # Keep the owning process alive until the final history write/lock release.
        while server.jobs.active_job() is not None:
            time.sleep(0.1)
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("Сервер ещё завершает работу. Дождитесь остановки приложения.")


def visible_error(text: str) -> None:
    """pythonw has no stderr: bootstrap failures must remain visible."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "Alfa Lab", 0x10)
    else:
        print(text, file=sys.stderr or sys.__stderr__)


def main() -> int:
    root = None
    server = None
    worker = None
    closer = None
    try:
        try:
            import tkinter as tk
            from tkinter import messagebox
        except ImportError as error:
            raise RuntimeError(
                "В Python отсутствует компонент tkinter. Добавьте Tcl/Tk через "
                "установщик Python 3.12 и повторите запуск."
            ) from error

        root = tk.Tk()
        root.title("Alfa Lab — управление приложением")
        root.geometry("560x400")
        root.minsize(520, 380)
        root.configure(bg="#f7f8fa")
        root.option_add("*Font", "{Segoe UI} 10")

        from app.server import WorkspaceServer

        data_root, state_root = project_paths()
        server = open_server(WorkspaceServer, data_root, state_root)
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                                  name="alfa-lab-server", daemon=False)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}/"

        panel = tk.Frame(root, bg="#ffffff", padx=28, pady=24)
        panel.pack(fill="both", expand=True, padx=16, pady=16)
        heading = tk.Frame(panel, bg="#ffffff")
        heading.pack(fill="x")
        tk.Label(heading, text="A", fg="#e5232e", bg="#ffffff",
                 font=("Segoe UI", 28, "bold")).pack(side="left", padx=(0, 12))
        tk.Label(heading, text="Alfa Lab", fg="#20232b", bg="#ffffff",
                 font=("Segoe UI", 21, "bold")).pack(side="left")
        status = tk.StringVar(value="● Лаборатория работает на этом компьютере")
        status_label = tk.Label(panel, textvariable=status, fg="#2b8266", bg="#ffffff",
                                anchor="w")
        status_label.pack(fill="x", pady=(13, 5))
        tk.Label(panel, text=url, fg="#667085", bg="#ffffff", anchor="w").pack(fill="x")
        tk.Label(panel, text="Работайте в браузере. Это окно можно свернуть.",
                 fg="#667085", bg="#ffffff", anchor="w").pack(fill="x", pady=(7, 17))
        buttons = tk.Frame(panel, bg="#ffffff")
        buttons.pack(fill="x")
        open_button = tk.Button(buttons, text="Открыть лабораторию", bg="#e5232e", fg="white",
                                activebackground="#cc1b26", activeforeground="white",
                                relief="flat", padx=14, pady=8,
                                command=lambda: webbrowser.open(url))
        open_button.pack(side="left")

        def copy_address():
            root.clipboard_clear()
            root.clipboard_append(url)
            root.update_idletasks()

        copy_button = tk.Button(buttons, text="Скопировать адрес", relief="flat",
                                bg="#f3f4f6", fg="#374151", padx=10, pady=8,
                                command=copy_address)
        copy_button.pack(side="left", padx=(8, 0))

        close_errors = []

        def stop_owned_server():
            try:
                close_server(server, worker)
            except Exception as error:
                close_errors.append(error)

        def poll_close():
            nonlocal closer
            if closer is not None and closer.is_alive():
                root.after(80, poll_close)
            elif close_errors:
                error = close_errors.pop()
                closer = None
                status.set("Остановка ещё не завершена")
                stop_button.configure(state="normal")
                messagebox.showerror("Не удалось завершить работу", str(error), parent=root)
            else:
                root.destroy()

        def request_close():
            nonlocal closer
            if closer is not None:
                return
            active = server.jobs.active_job()
            if active and not messagebox.askyesno(
                "Завершить работу?",
                "Сейчас выполняется эксперимент. Остановить его и закрыть лабораторию?",
                parent=root,
            ):
                return
            status.set("Завершаем работу и сохраняем историю…")
            status_label.configure(fg="#667085")
            for button in (open_button, copy_button, stop_button):
                button.configure(state="disabled")
            closer = threading.Thread(target=stop_owned_server,
                                      name="alfa-lab-stop", daemon=False)
            closer.start()
            poll_close()

        stop_button = tk.Button(panel, text="Завершить работу", relief="flat", bg="#ffffff",
                                fg="#667085", command=request_close)
        stop_button.pack(anchor="w", pady=(12, 0))
        root.protocol("WM_DELETE_WINDOW", request_close)
        root.after(150, lambda: webbrowser.open(url))
        root.mainloop()
        return 0
    except Exception as error:
        if isinstance(error, ModuleNotFoundError) and error.name != "tkinter":
            detail = ("В окружении проекта отсутствуют зависимости. Установите requirements.txt "
                      "в .venv и повторите запуск.\n\nОтсутствует модуль: " + str(error.name))
        else:
            detail = str(error) or type(error).__name__
        visible_error("Не удалось открыть лабораторию.\n\n" + detail)
        return 1
    finally:
        if closer is not None:
            closer.join()
        close_server(server, worker)
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
