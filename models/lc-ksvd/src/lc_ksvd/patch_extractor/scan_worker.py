"""
scan_worker.py
Runs the per-scan abnormal-sampling step in an isolated, persistent
subprocess so a native-library segfault or interpreter-level fault (e.g.
the SystemError/segfault observed from json.dumps under memory pressure,
or a lungmask/nibabel crash) only loses one scan's work, never the whole
multi-hour pass.

The parent process owns all patch writing and checkpointing; the worker
only computes and returns per-scan (labels, coords, patches) over a Queue.
"""

import logging
import multiprocessing as mp
import queue
from collections.abc import Callable

import numpy as np

from lc_ksvd.config import CLASS_ORDER, PATCH_SIZE
from lc_ksvd.data_loader.metadata_registry import MetadataRegistry
from lc_ksvd.data_loader.scan_loader import ScanLoader

logger = logging.getLogger(__name__)

_CTX = mp.get_context("spawn")

ScanProcessFn = Callable[
    [str, ScanLoader, dict],
    tuple[list[int], list[tuple[int, int, int]], np.ndarray],
]


def _worker_main(task_q, result_q, metadata: MetadataRegistry, process_fn: ScanProcessFn) -> None:
    loader = ScanLoader(metadata)
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}
    while True:
        scan_id = task_q.get()
        if scan_id is None:
            break
        try:
            labels, coords, patches = process_fn(scan_id, loader, class_to_idx)
            result_q.put((scan_id, "ok", labels, coords, patches))
        except Exception as exc:
            logger.warning(f"  {scan_id}: worker raised {exc!r}")
            result_q.put((scan_id, "error", str(exc), None, None))


class ScanWorkerPool:
    """
    Single persistent worker subprocess. If it dies (segfault) or a scan
    exceeds `timeout_s`, the pool discards it and starts a fresh one; the
    caller gets one retry on the new worker before the scan is recorded as
    zero patches and skipped.
    """

    def __init__(self, metadata: MetadataRegistry, process_fn: ScanProcessFn, timeout_s: float = 60.0):
        self._metadata = metadata
        self._process_fn = process_fn
        self._timeout_s = timeout_s
        self._task_q: mp.Queue | None = None
        self._result_q: mp.Queue | None = None
        self._proc: mp.process.BaseProcess | None = None
        self._start()

    def _start(self) -> None:
        self._task_q = _CTX.Queue()
        self._result_q = _CTX.Queue()
        self._proc = _CTX.Process(
            target=_worker_main,
            args=(self._task_q, self._result_q, self._metadata, self._process_fn),
            daemon=True,
        )
        self._proc.start()

    def _kill(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
        for q in (self._task_q, self._result_q):
            if q is not None:
                q.close()

    def _restart(self) -> None:
        logger.warning("Worker died or timed out — restarting a fresh worker process.")
        self._kill()
        self._start()

    def _await_result(self, poll_interval: float = 1.0):
        """Poll for a result, failing fast (within ~poll_interval) if the
        worker process has already died, instead of blocking the full
        timeout on a process that will never respond."""
        waited = 0.0
        while waited < self._timeout_s:
            try:
                return self._result_q.get(timeout=poll_interval)
            except queue.Empty:
                waited += poll_interval
                if not self._proc.is_alive():
                    return None
        return None

    def process(self, scan_id: str) -> tuple[list[int], list[tuple[int, int, int]], np.ndarray]:
        for attempt in (1, 2):
            self._task_q.put(scan_id)
            result = self._await_result()

            if result is None:
                logger.warning(f"  {scan_id}: worker died or timed out (attempt {attempt}).")
                self._restart()
                continue

            got_id, status, a, b, c = result
            if status == "ok":
                assert got_id == scan_id
                return a, b, c

            logger.warning(f"  {scan_id}: {a} (attempt {attempt}).")
            self._restart()

        logger.warning(f"  {scan_id}: failed twice — recording 0 patches and moving on.")
        return [], [], np.empty((0, *PATCH_SIZE), dtype=np.float32)

    def shutdown(self) -> None:
        try:
            if self._task_q is not None:
                self._task_q.put(None)
            if self._proc is not None:
                self._proc.join(timeout=10)
        finally:
            self._kill()

    def __enter__(self) -> "ScanWorkerPool":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()