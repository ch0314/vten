"""POSIX Named Semaphore wrapper.

Thin ctypes wrapper around sem_open/sem_post/sem_wait/sem_timedwait.
Falls back gracefully when POSIX APIs are unavailable (non-Linux, etc.).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time


class PosixSemaphore:
    """Thin ctypes wrapper around POSIX named semaphores.

    Falls back gracefully when POSIX APIs are unavailable (non-Linux, etc.).
    """

    _lib = None
    _available = None

    @classmethod
    def _load(cls) -> bool:
        if cls._available is not None:
            return cls._available
        for libname in ("pthread", "rt", "c"):
            path = ctypes.util.find_library(libname)
            if path:
                try:
                    cls._lib = ctypes.CDLL(path, use_errno=True)
                    if hasattr(cls._lib, "sem_open"):
                        cls._available = True
                        return True
                except OSError:
                    continue
        cls._available = False
        return False

    def __init__(self, name: str, *, create: bool = False) -> None:
        self._name = name.encode("utf-8")
        self._sem = None

        if not self._load():
            return

        lib = self._lib
        lib.sem_open.restype = ctypes.c_void_p
        if create:
            self._sem = lib.sem_open(
                self._name, ctypes.c_int(os.O_CREAT), ctypes.c_uint(0o644), ctypes.c_uint(0),
            )
        else:
            self._sem = lib.sem_open(self._name, ctypes.c_int(0))

        SEM_FAILED = ctypes.c_void_p(-1).value
        if self._sem is None or self._sem == SEM_FAILED:
            self._sem = None

    def post(self) -> None:
        if self._sem is None:
            return
        self._lib.sem_post(ctypes.c_void_p(self._sem))

    def wait(self) -> bool:
        if self._sem is None:
            return True
        return self._lib.sem_wait(ctypes.c_void_p(self._sem)) == 0

    def timedwait(self, timeout_s: float) -> bool:
        """Wait with timeout. Returns True if acquired, False on timeout."""
        if self._sem is None:
            return True

        if not hasattr(self._lib, "sem_timedwait"):
            return self.wait()

        class _timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        deadline = time.time() + timeout_s
        ts = _timespec()
        ts.tv_sec = int(deadline)
        ts.tv_nsec = int((deadline - int(deadline)) * 1_000_000_000)

        result = self._lib.sem_timedwait(ctypes.c_void_p(self._sem), ctypes.byref(ts))
        return result == 0

    def close(self) -> None:
        if self._sem is None:
            return
        try:
            self._lib.sem_close(ctypes.c_void_p(self._sem))
        except Exception:
            pass
        self._sem = None

    def unlink(self) -> None:
        try:
            self._lib.sem_unlink(self._name)
        except Exception:
            pass
