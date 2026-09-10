"""
Structured logging so every module reports real, non-fabricated numbers
(Section 74): input rows, output rows, parameters, runtime, artifact location.

Usage:
    from src.logging_config import get_module_logger, ModuleRun

    log = get_module_logger("MODULE_03_SEQUENTIAL_MINING")
    with ModuleRun(log, module="MODULE 03") as run:
        ... do work, then:
        run.record(input_rows=len(df), output_rows=len(patterns),
                   artifact_location=str(out_path))
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_module_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class ModuleRun(AbstractContextManager):
    """Context manager that times a module run and logs a structured summary.
    Never hard-codes fake numbers — every field must be supplied via .record()
    with values actually computed during the run.
    """

    logger: logging.Logger
    module: str
    _start: float = field(default=0.0, init=False)
    _fields: dict[str, Any] = field(default_factory=dict, init=False)
    _error: Exception | None = field(default=None, init=False)

    def __enter__(self) -> "ModuleRun":
        self._start = time.time()
        self.logger.info(f"[{self.module}] START")
        return self

    def record(self, **kwargs: Any) -> None:
        self._fields.update(kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        runtime = time.time() - self._start
        if exc_type is not None:
            self.logger.error(f"[{self.module}] FAILED after {runtime:.2f}s — {exc_val}")
            return False  # re-raise; never swallow errors silently
        summary = " | ".join(f"{k}={v}" for k, v in self._fields.items())
        self.logger.info(f"[{self.module}] COMPLETE | runtime={runtime:.2f}s | {summary}")
        return False
