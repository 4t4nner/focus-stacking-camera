#!/usr/bin/env python3
"""
Автоматический прогон валидационных сцен focus stacking.
Отправляет серии на сервер, собирает реальные метрики для главы 3.5:
  - тайминги стадий (из server_pipeline_log_*.json)
  - дисперсия лапласиана входов и результата (К3)
  - доля unfocused-пикселей (К4)
  - число уникальных кадров (filter_duplicates)
Результат -> validation_results.csv + печать в консоль.
"""
import os, sys, glob, time, json, csv
import requests
import cv2
import numpy as np

SERVER = "http://127.0.0.1:5000"
# Корень debug_output сервера (где он создаёт session_*).
# Скрипт читает свежесозданную сессию для метрик по артефактам.
DEBUG_ROOT = "./debug_output"

# Каждая сцена = папка со своими кадрами серии (одинакового разрешения!).
# Структура: scenes/S1/*.jpg, scenes/S2/*.jpg ...
SCENES_DIR = "./scenes"


def sharpness(path_or_img):
    img = (cv2.imread(path_or_img, cv2.IMREAD_GRAYSCALE)
           if isinstance(path_or_img, str) else path_or_img)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def newest_session_after(t0):
    """Найти session_*, созданную после t0."""
    sessions = glob.glob(os.path.join(DEBUG_ROOT, "session_*"))
    sessions = [s for s in sessions if os.path.getmtime(s) >= t0 - 1]
    if not sessions:
        return None
    return max(sessions, key=os.path.getmtime)


def submit_scene(image_paths):
    files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg"))
             for p in image_paths]
    r = requests.post(f"{SERVER}/api/focus-stack", files=files)
    for _, (_, fh, _) in files:
        fh.close()
    r.raise_for_status()
    return r.json()["job_id"]


def wait_done(job_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.get(f"{SERVER}/api/status/{job_id}").json()
        if r["status"] == "done":
            return True
        if r["status"] == "error":
            raise RuntimeError(f"Job error: {r.get('error')}")
        time.sleep(1.0)
    raise TimeoutError(f"Job {job_id} timeout")


def metrics_from_session(session_dir, input_paths):
    out = {}

    # --- тайминги ---
    logs = glob.glob(os.path.join(session_dir, "server_pipeline_log_*.json"))
    if logs:
        with open(max(logs, key=os.path.getmtime)) as f:
            plog = json.load(f)
        out["total_ms"] = plog.get("total_duration_ms")
        for st in plog.get("pipeline_stages", []):
            out[f"stage_{st['stage']}_ms"] = st["duration_ms"]
        out["image_size"] = plog.get("meta", {}).get("image_size")

    # --- уникальные кадры ---
    uj = os.path.join(session_dir, "unique_indices.json")
    if os.path.exists(uj):
        with open(uj) as f:
            u = json.load(f)
        out["n_total"] = u["total"]
        out["n_unique"] = len(u["unique_indices"])

    # --- К3: дисперсия лапласиана ---
    s_inputs = [sharpness(p) for p in input_paths]
    result_path = os.path.join(session_dir, "composite_final.jpg")
    out["sharp_mean_input"] = round(float(np.mean(s_inputs)), 1)
    out["sharp_max_input"] = round(float(np.max(s_inputs)), 1)
    if os.path.exists(result_path):
        s_res = sharpness(result_path)
        out["sharp_result"] = round(s_res, 1)
        out["gain_vs_mean"] = round(s_res / np.mean(s_inputs), 3)
        out["gain_vs_max"] = round(s_res / np.max(s_inputs), 3)

    # --- К4: целостность карты источников ---
    fmap = os.path.join(session_dir, "focusmap_raw.png")
    if os.path.exists(fmap):
        fm = cv2.imread(fmap, cv2.IMREAD_GRAYSCALE)
        # zone_mask_*.png дают точное число назначенных пикселей по зонам
        zone_masks = sorted(glob.glob(os.path.join(session_dir, "zone_mask_*.png")))
        if zone_masks:
            assigned = np.zeros(fm.shape, dtype=bool)
            for zm in zone_masks:
                m = cv2.imread(zm, cv2.IMREAD_GRAYSCALE) > 127
                assigned |= m
            unfocused = np.count_nonzero(~assigned)
            out["unfocused_pct"] = round(100.0 * unfocused / fm.size, 4)
        out["n_zones"] = len(zone_masks)

    return out


def main():
    scenes = sorted(glob.glob(os.path.join(SCENES_DIR, "*")))
    scenes = [s for s in scenes if os.path.isdir(s)]
    if not scenes:
        print(f"Нет сцен в {SCENES_DIR}/<scene>/*.jpg")
        sys.exit(1)

    rows = []
    for scene in scenes:
        name = os.path.basename(scene)
        imgs = sorted(glob.glob(os.path.join(scene, "*.jpg")) +
                      glob.glob(os.path.join(scene, "*.png")))
        if len(imgs) < 2:
            print(f"[{name}] пропущена (нужно >=2 кадра)")
            continue

        print(f"[{name}] отправка {len(imgs)} кадров...")
        t0 = time.time()
        job_id = submit_scene(imgs)
        wait_done(job_id)
        session = newest_session_after(t0)
        if session is None:
            print(f"[{name}] session_dir не найден")
            continue

        m = metrics_from_session(session, imgs)
        m["scene"] = name
        m["job_id"] = job_id
        rows.append(m)
        print(f"[{name}] gain_vs_max={m.get('gain_vs_max')} "
              f"unfocused={m.get('unfocused_pct')}% "
              f"total={m.get('total_ms')}ms")

    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open("validation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nСохранено: validation_results.csv")


if __name__ == "__main__":
    main()

