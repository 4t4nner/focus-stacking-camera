#!/usr/bin/env python3
"""
Aggregate server pipeline logs into a summary table.

Usage:
    python aggregate_logs.py ./debug_output
    python aggregate_logs.py ./debug_output --csv summary.csv --stages-csv stages.csv
"""

import os
import csv
import glob
import json
import argparse
import statistics
from collections import defaultdict


def find_log_files(root: str):
    """Найти все server_pipeline_log_*.json рекурсивно."""
    pattern = os.path.join(root, "**", "server_pipeline_log_*.json")
    return sorted(glob.glob(pattern, recursive=True))


def load_log(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Не удалось прочитать {path}: {e}")
        return None


def stages_to_dict(log: dict):
    """Список стадий -> {stage: duration_ms} (при дублях суммируем)."""
    d = defaultdict(float)
    for s in log.get("pipeline_stages", []):
        d[s["stage"]] += s.get("duration_ms", 0.0)
    return dict(d)


def build_session_row(log: dict, path: str):
    meta = log.get("meta", {})
    summary = log.get("summary", {})
    return {
        "job_id": log.get("job_id", ""),
        "session_dir": os.path.basename(os.path.dirname(path)),
        "n_files": meta.get("n_files", ""),
        "n_unique": meta.get("n_unique", ""),
        "image_size": meta.get("image_size", ""),
        "yolo_available": meta.get("yolo_available", ""),
        "total_duration_ms": log.get("total_duration_ms", ""),
        "total_stage_ms": summary.get("total_stage_ms", ""),
        "peak_rss_mb": summary.get("peak_rss_mb", ""),
        "peak_gpu_allocated_mb": summary.get("peak_gpu_allocated_mb", ""),
        "peak_gpu_max_allocated_mb": summary.get("peak_gpu_max_allocated_mb", ""),
        "cuda_available": log.get("cuda_available", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Каталог debug_output (или родительский)")
    ap.add_argument("--csv", default="server_logs_summary.csv",
                    help="Файл сводки по сессиям")
    ap.add_argument("--stages-csv", default="server_logs_stages.csv",
                    help="Файл сводки по стадиям (среднее/медиана)")
    args = ap.parse_args()

    files = find_log_files(args.root)
    if not files:
        print(f"Логи не найдены в {args.root}")
        return

    print(f"Найдено логов: {len(files)}\n")

    session_rows = []
    stage_durations = defaultdict(list)  # stage -> [ms, ...]
    all_stage_names = []                 # для стабильного порядка колонок

    for path in files:
        log = load_log(path)
        if log is None:
            continue
        session_rows.append(build_session_row(log, path))
        for st, dur in stages_to_dict(log).items():
            if st not in stage_durations:
                all_stage_names.append(st)
            stage_durations[st].append(dur)

    # ---- сводка по сессиям ----
    session_fields = list(session_rows[0].keys())
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=session_fields)
        wr.writeheader()
        wr.writerows(session_rows)
    print(f"[OK] Сводка по сессиям -> {args.csv} ({len(session_rows)} строк)")

    # ---- сводка по стадиям ----
    stage_summary = []
    for st in all_stage_names:
        vals = stage_durations[st]
        stage_summary.append({
            "stage": st,
            "count": len(vals),
            "mean_ms": round(statistics.mean(vals), 1),
            "median_ms": round(statistics.median(vals), 1),
            "min_ms": round(min(vals), 1),
            "max_ms": round(max(vals), 1),
            "stdev_ms": round(statistics.stdev(vals), 1) if len(vals) > 1 else 0.0,
        })

    with open(args.stages_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(stage_summary[0].keys()))
        wr.writeheader()
        wr.writerows(stage_summary)
    print(f"[OK] Сводка по стадиям -> {args.stages_csv}\n")

    # ---- печать в консоль ----
    print("=== Стадии (среднее по всем сессиям) ===")
    print(f"{'stage':<26}{'count':>6}{'mean_ms':>10}{'median':>10}{'max_ms':>10}")
    for r in sorted(stage_summary, key=lambda x: -x["mean_ms"]):
        print(f"{r['stage']:<26}{r['count']:>6}{r['mean_ms']:>10}"
              f"{r['median_ms']:>10}{r['max_ms']:>10}")

    # агрегаты по памяти/времени сессий
    def col(name):
        return [r[name] for r in session_rows
                if isinstance(r[name], (int, float))]

    rss = col("peak_rss_mb")
    gpu = col("peak_gpu_max_allocated_mb")
    dur = col("total_duration_ms")

    print("\n=== Память и время (по сессиям) ===")
    if dur:
        print(f"total_duration_ms : mean={statistics.mean(dur):.0f}  "
              f"max={max(dur):.0f}")
    if rss:
        print(f"peak_rss_mb       : mean={statistics.mean(rss):.0f}  "
              f"max={max(rss):.0f}")
    if gpu:
        print(f"peak_gpu_max_mb   : mean={statistics.mean(gpu):.0f}  "
              f"max={max(gpu):.0f}")


if __name__ == "__main__":
    main()

