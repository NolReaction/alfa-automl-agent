"""Temporary Windows idle-sleep request owned by this application's worker.

No power plan, display setting or global registry value is changed. Windows
associates SetThreadExecutionState with its calling thread, so set and release
always happen on the same dedicated thread.
"""
from __future__ import annotations

import sys
import threading

CONTINUOUS = 0x80000000
SYSTEM_REQUIRED = 0x00000001


class SleepGuard:
    def __init__(self, activity, *, platform=None, setter=None, autostart=True):
        self._activity, self._setter = activity, setter
        self._supported = (platform or sys.platform) == "win32"
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._requested = self._active = False
        self._reason = ("Ожидание работы. Сон дисплея разрешён." if self._supported else
                        "Автоматическое предотвращение сна доступно только в Windows.")
        self._thread = None
        if self._supported and autostart:
            self._thread = threading.Thread(target=self._run, name="alfa-idle-sleep", daemon=True)
            self._thread.start()

    def _call(self, flags):
        if self._setter is None:
            import ctypes
            function = ctypes.windll.kernel32.SetThreadExecutionState
            function.argtypes = [ctypes.c_uint32]
            function.restype = ctypes.c_uint32
            self._setter = function
        return bool(self._setter(flags))

    def _tick(self, requested):
        with self._lock:
            self._requested = bool(requested)
            if not self._supported:
                return
            if self._active == self._requested:
                return
            try:
                accepted = self._call(CONTINUOUS | (SYSTEM_REQUIRED if requested else 0))
            except (OSError, AttributeError):
                accepted = False
            if accepted:
                self._active = self._requested
                self._reason = ("Windows приняла запрос не засыпать при простое во время работы. "
                                "Он не защищает от закрытия крышки, ручного сна или перезагрузки."
                                if requested else "Работы нет. Обычные настройки сна восстановлены.")
            else:
                self._reason = "Windows не подтвердила запрос. Проверьте настройки сна перед долгим запуском."

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    self._tick(bool(self._activity()))
                except Exception:
                    # Keep the last owned request until activity is observable;
                    # do not silently declare that the training has finished.
                    with self._lock:
                        self._reason = "Не удалось проверить активную работу; состояние сна требует проверки."
                self._stop.wait(2)
        finally:
            self._tick(False)

    def snapshot(self):
        with self._lock:
            return {"supported": self._supported, "requested": self._requested,
                    "active": self._active, "reason": self._reason}

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
