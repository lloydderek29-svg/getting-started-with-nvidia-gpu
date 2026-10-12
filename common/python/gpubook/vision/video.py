from __future__ import annotations

import queue
import threading
import time

import cv2


class FrameSource:
    """Read frames on a thread.

    With drop_stale=True (cameras), a slow consumer always gets
    the newest frame and old ones are dropped. With False
    (files), the reader waits so that no frame is skipped.
    """

    def __init__(self, source, drop_stale: bool = True):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open source {source!r}")
        self.drop_stale = drop_stale
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True)
        self._thread.start()

    def _put(self, item) -> None:
        if not self.drop_stale:
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def _run(self) -> None:
        index = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            stamp = time.perf_counter()
            if not ok:
                self._put(None)
                return
            self._put((frame, stamp, index))
            index += 1

    def read(self, timeout: float = 2.0):
        """Return (frame, capture_time, index) or None at end."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()
