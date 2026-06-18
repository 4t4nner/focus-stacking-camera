#!/usr/bin/env python3
"""
Server-side pipeline logger: stage timings + RAM/GPU memory snapshots.
Mirrors the client-side PipelineLogger so both JSON logs can be compared.
"""

import os
import json
import time
from datetime import datetime
from contextlib import contextmanager

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    torch = None
    _HAS_CUDA = False


class ServerPipelineLogger:
    """One instance per job. Collects stage durations and memory snapshots."""

    def __init__(self, session_dir: str, job_id: str):
        self.session_dir = session_dir
        self.job_id = job_id
        self.session_start = time.time()
        self.stages = []            # [{stage, duration_ms}]
        self.memory_snapshots = []  # [{tag, ...}]
        self.meta = {}
        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        if _HAS_CUDA:
            # сброс пикового счётчика, чтобы измерять пик именно этой сессии
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def set_meta(self, **kwargs):
        self.meta.update(kwargs)

    @contextmanager
    def measure_stage(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            dur_ms = (time.perf_counter() - start) * 1000.0
            self.stages.append({"stage": stage, "duration_ms": round(dur_ms, 2)})
            print(f"[Job {self.job_id}]   [STAGE] {stage}: {dur_ms:.1f}ms",
                  flush=True)

    def log_memory(self, tag: str):
        snap = {"tag": tag}

        if self._proc is not None:
            mem = self._proc.memory_info()
            vm = psutil.virtual_memory()
            snap.update({
                "rss_mb": round(mem.rss / (1024 ** 2), 1),     # фактическая RAM процесса
                "vms_mb": round(mem.vms / (1024 ** 2), 1),     # виртуальная память
                "host_total_mb": round(vm.total / (1024 ** 2), 1),
                "host_avail_mb": round(vm.available / (1024 ** 2), 1),
                "host_used_percent": vm.percent,
            })

        if _HAS_CUDA:
            snap.update({
                "gpu_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 1),
                "gpu_reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 1),
                "gpu_max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
            })

        self.memory_snapshots.append(snap)

        rss_str = f"RSS={snap.get('rss_mb', '?')}MB" if self._proc else "RSS=n/a"
        gpu_str = (f", GPU={snap['gpu_allocated_mb']:.1f}MB"
                   f"(peak {snap['gpu_max_allocated_mb']:.1f})") if _HAS_CUDA else ""
        host_str = (f", host avail={snap['host_avail_mb']:.0f}/"
                    f"{snap['host_total_mb']:.0f}MB") if self._proc else ""
        print(f"[Job {self.job_id}]   [MEMORY {tag}]: {rss_str}{host_str}{gpu_str}",
              flush=True)
        return snap

    def write_json(self) -> str:
        # --- агрегаты по памяти ---
        rss_vals = [s["rss_mb"] for s in self.memory_snapshots if "rss_mb" in s]
        gpu_vals = [s["gpu_allocated_mb"] for s in self.memory_snapshots
                    if "gpu_allocated_mb" in s]
        gpu_peak_vals = [s["gpu_max_allocated_mb"] for s in self.memory_snapshots
                         if "gpu_max_allocated_mb" in s]

        summary = {
            "peak_rss_mb": round(max(rss_vals), 1) if rss_vals else None,
            "peak_gpu_allocated_mb": round(max(gpu_vals), 1) if gpu_vals else None,
            # пик за сессию по данным torch (надёжнее, чем max по снимкам)
            "peak_gpu_max_allocated_mb": round(max(gpu_peak_vals), 1) if gpu_peak_vals else None,
            "total_stage_ms": round(sum(s["duration_ms"] for s in self.stages), 2),
        }

        root = {
            "job_id": self.job_id,
            "session_start": self.session_start,
            "session_end": time.time(),
            "total_duration_ms": round((time.time() - self.session_start) * 1000.0, 2),
            "psutil_available": _HAS_PSUTIL,
            "cuda_available": _HAS_CUDA,
            "summary": summary,
            "meta": self.meta,
            "pipeline_stages": self.stages,
            "memory_snapshots": self.memory_snapshots,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.session_dir, f"server_pipeline_log_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
            print(f"[Job {self.job_id}]   [DEBUG] Pipeline log written: {path} "
                  f"(peak RSS={summary['peak_rss_mb']}MB)", flush=True)
        except Exception as e:
            print(f"[Job {self.job_id}]   Failed to write pipeline log: {e}",
                  flush=True)
        return path

