#!/usr/bin/env python3
"""
Focus Stacking Remote Server — Async Job Queue with YOLO refinement.
"""
from pipeline_logger import ServerPipelineLogger
import os
import io
import re
import tempfile
import traceback
import time
import json
import uuid
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# ======================== CONFIG ========================

DEBUG_OUTPUT_DIR = os.environ.get("DEBUG_OUTPUT_DIR", "./debug_output")

# Job storage
jobs = {}
jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 600

# YOLO availability flag (set at startup)
_yolo_available = False


def get_session_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = os.path.join(DEBUG_OUTPUT_DIR, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def save_debug_image(session_dir: str, filename: str, image: np.ndarray,
                     is_mask: bool = False):
    path = os.path.join(session_dir, filename)
    if is_mask:
        cv2.imwrite(path, image)
    else:
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    app.logger.info(f"  [DEBUG] Saved: {path}")


def save_focus_map_visualization(session_dir: str, focus_map: np.ndarray,
                                 n_sources: int, focus_points: list):
    h, w = focus_map.shape
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255),
        (128, 255, 0),
    ]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n_sources):
        mask = focus_map == i
        color = colors[i % len(colors)]
        vis[mask] = color
    for i, fp in enumerate(focus_points):
        cx = int(np.clip(fp[0], 0, w - 1))
        cy = int(np.clip(fp[1], 0, h - 1))
        cv2.circle(vis, (cx, cy), 15, (255, 255, 255), 3)
        cv2.circle(vis, (cx, cy), 12, colors[i % len(colors)], -1)
        cv2.putText(vis, str(i), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(session_dir, "focusmap_color.png"), vis)
    app.logger.info(f"  [DEBUG] Saved focus map visualization: {session_dir}/focusmap_color.png")
    if n_sources > 1:
        raw_vis = (focus_map.astype(np.float32) / (n_sources - 1) * 255).astype(np.uint8)
    else:
        raw_vis = np.zeros_like(focus_map, dtype=np.uint8)
    cv2.imwrite(os.path.join(session_dir, "focusmap_raw.png"), raw_vis)
    app.logger.info(f"  [DEBUG] Saved raw focus map: {session_dir}/focusmap_raw.png")


def save_annotated_image(session_dir: str, image: np.ndarray,
                         focus_map: np.ndarray, n_sources: int,
                         focus_points: list):
    h, w = focus_map.shape
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255),
        (128, 255, 0),
    ]
    overlay = image.copy()
    zone_overlay = np.zeros_like(image)
    for i in range(n_sources):
        mask = focus_map == i
        color = colors[i % len(colors)]
        zone_overlay[mask] = color
    cv2.addWeighted(zone_overlay, 0.25, overlay, 1.0, 0, overlay)
    padded = np.pad(focus_map, 1, mode='edge')
    seam_mask = np.zeros((h, w), dtype=np.uint8)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            seam_mask |= (focus_map != shifted).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seam_mask = cv2.dilate(seam_mask, kernel)
    overlay[seam_mask > 0] = (0, 255, 255)
    for i, fp in enumerate(focus_points):
        cx = int(np.clip(fp[0], 0, w - 1))
        cy = int(np.clip(fp[1], 0, h - 1))
        cv2.circle(overlay, (cx, cy), 20, (255, 255, 255), 3)
        cv2.circle(overlay, (cx, cy), 17, colors[i % len(colors)], -1)
        cv2.putText(overlay, f"F{i}", (cx - 10, cy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    path = os.path.join(session_dir, "annotated_composite.jpg")
    cv2.imwrite(path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    app.logger.info(f"  [DEBUG] Saved annotated composite: {path}")


def cleanup_old_jobs():
    now = time.time()
    with jobs_lock:
        expired = [
            jid for jid, jdata in jobs.items()
            if now - jdata.get("created_at", now) > JOB_TTL_SECONDS
               and jdata["status"] in ("done", "error")
        ]
        for jid in expired:
            jobs.pop(jid, None)


# ======================== PIPELINE WORKER ========================

def process_job(job_id: str, image_data_list: list, focus_points: list,
                threshold: int, kernel_size: int, sigma1: float,
                sigma2: float, gain: float, filenames: list):
    """Background processing thread."""
    plog = None
    try:
        session_dir = get_session_dir()

        # --- pipeline logger ---
        plog = ServerPipelineLogger(session_dir, job_id)
        plog.set_meta(
            n_files=len(image_data_list),
            n_focus_points=len(focus_points),
            threshold=threshold,
            kernel_size=kernel_size,
            sigma1=sigma1,
            sigma2=sigma2,
            gain=gain,
            yolo_available=_yolo_available,
        )
        plog.log_memory("request_start")

        with jobs_lock:
            jobs[job_id]["session_dir"] = session_dir
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["step"] = "saving_inputs"

        app.logger.info(f"[Job {job_id}] Session dir: {session_dir}")

        params = {
            "job_id": job_id,
            "n_files": len(image_data_list),
            "focus_points": focus_points,
            "threshold": threshold,
            "kernel_size": kernel_size,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "gain": gain,
            "filenames": filenames,
            "yolo_available": _yolo_available,
        }
        with open(os.path.join(session_dir, "params.json"), "w") as pf:
            json.dump(params, pf, indent=2)

        # Save input images
        with plog.measure_stage("save_inputs"):
            image_paths = []
            for i, img_bytes in enumerate(image_data_list):
                input_path = os.path.join(session_dir, f"input_{i:03d}.jpg")
                with open(input_path, "wb") as f:
                    f.write(img_bytes)
                image_paths.append(input_path)
                app.logger.info(f"[Job {job_id}]   Image {i}: {filenames[i]} saved")

        # Step 1: Detail masks (DoG)
        with jobs_lock:
            jobs[job_id]["step"] = "generating_masks"

        with plog.measure_stage("generate_masks"):
            masks = []
            for i, img_path in enumerate(image_paths):
                mask = generate_details_mask(
                    img_path, threshold, kernel_size, sigma1, sigma2, gain
                )
                masks.append(mask)
                save_debug_image(session_dir, f"details_{i:03d}.png", mask, is_mask=True)
                app.logger.info(f"[Job {job_id}]   Mask {i}: shape={mask.shape}")

        if masks:
            mh, mw = masks[0].shape[:2]
            plog.set_meta(image_size=f"{mw}x{mh}")
        plog.log_memory("after_masks")

        # Step 1b: Filter duplicates
        with plog.measure_stage("filter_duplicates"):
            unique_indices = filter_duplicate_masks(masks, similarity_threshold=0.05)
        app.logger.info(f"[Job {job_id}]   Unique masks: {unique_indices} / {len(masks)}")
        plog.set_meta(n_unique=len(unique_indices))

        with open(os.path.join(session_dir, "unique_indices.json"), "w") as uf:
            json.dump({"unique_indices": unique_indices, "total": len(masks)}, uf)

        if len(unique_indices) < 2:
            result_path = os.path.join(session_dir, "composite_final.jpg")
            shutil.copy2(image_paths[0], result_path)
            plog.log_memory("complete")
            plog.write_json()
            with jobs_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["result_path"] = result_path
                jobs[job_id]["step"] = "complete"
            return

        u_image_paths = [image_paths[i] for i in unique_indices]
        u_masks = [masks[i] for i in unique_indices]
        u_focus_points = [
            focus_points[i] if i < len(focus_points)
            else {"cx": masks[i].shape[1] // 2, "cy": masks[i].shape[0] // 2}
            for i in unique_indices
        ]

        # Step 2: Align
        with jobs_lock:
            jobs[job_id]["step"] = "aligning"

        app.logger.info(f"[Job {job_id}]   Aligning images...")
        with plog.measure_stage("align_images"):
            aligned_images, aligned_masks, homographies = align_images(
                u_image_paths, u_masks
            )
        app.logger.info(f"[Job {job_id}]   Aligned: {len(aligned_images)} images")
        plog.log_memory("after_align")

        with plog.measure_stage("save_aligned_debug"):
            for i, (aimg, amask) in enumerate(zip(aligned_images, aligned_masks)):
                save_debug_image(session_dir, f"aligned_{i:03d}.jpg", aimg)
                save_debug_image(session_dir, f"aligned_mask_{i:03d}.png", amask,
                                 is_mask=True)

        # Step 3: Focus map
        with jobs_lock:
            jobs[job_id]["step"] = "building_focus_map"

        app.logger.info(f"[Job {job_id}]   Building focus map...")
        fp_list = [(fp["cx"], fp["cy"]) for fp in u_focus_points]
        with plog.measure_stage("build_focus_map"):
            focus_map = build_focus_map(aligned_masks, fp_list)
        app.logger.info(f"[Job {job_id}]   Focus map built: shape={focus_map.shape}")

        save_focus_map_visualization(
            session_dir, focus_map, len(aligned_images), fp_list
        )

        # Step 3b: YOLO refinement — keep objects intact
        with jobs_lock:
            jobs[job_id]["step"] = "yolo_refinement"

        with plog.measure_stage("yolo_refinement"):
            focus_map = _apply_yolo_refinement(
                job_id, session_dir, aligned_images[0], focus_map
            )
        plog.log_memory("after_yolo")

        # Step 4: Composite
        with jobs_lock:
            jobs[job_id]["step"] = "compositing"

        app.logger.info(f"[Job {job_id}]   Compositing...")
        with plog.measure_stage("composite"):
            composite = composite_images(aligned_images, focus_map)
        app.logger.info(f"[Job {job_id}]   Composite: shape={composite.shape}")
        save_debug_image(session_dir, "composite_raw.jpg", composite)

        # Step 5: Blend seams
        with jobs_lock:
            jobs[job_id]["step"] = "blending_seams"

        with plog.measure_stage("blend_seams"):
            composite = blend_seams(composite, aligned_images, focus_map)

        # Save results
        with plog.measure_stage("save_result"):
            result_path = os.path.join(session_dir, "composite_final.jpg")
            cv2.imwrite(result_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
            app.logger.info(f"[Job {job_id}]   [DEBUG] Saved: {result_path}")

            save_annotated_image(
                session_dir, composite, focus_map,
                len(aligned_images), fp_list
            )

        result_size = os.path.getsize(result_path)
        app.logger.info(f"[Job {job_id}]   Done! Composite size: {result_size} bytes")
        app.logger.info(f"[Job {job_id}]   [DEBUG] All intermediate files in: {session_dir}")

        plog.log_memory("complete")
        plog.write_json()

        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result_path"] = result_path
            jobs[job_id]["step"] = "complete"

    except Exception as e:
        traceback.print_exc()
        if plog is not None:
            plog.log_memory("error")
            plog.write_json()
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["step"] = "failed"
        app.logger.error(f"[Job {job_id}]   ERROR: {e}")

def _apply_yolo_refinement(
    job_id: str,
    session_dir: str,
    reference_image: np.ndarray,
    focus_map: np.ndarray,
) -> np.ndarray:
    """Run YOLO segmentation on reference image and refine focus map."""
    if not _yolo_available:
        app.logger.info(f"[Job {job_id}]   YOLO not available, skipping refinement")
        return focus_map

    try:
        from yolo_segmenter import segment_objects, refine_focus_map_with_objects

        app.logger.info(f"[Job {job_id}]   Running YOLO segmentation...")
        objects = segment_objects(reference_image)

        if not objects:
            app.logger.info(f"[Job {job_id}]   No objects detected by YOLO")
            return focus_map

        app.logger.info(
            f"[Job {job_id}]   YOLO found {len(objects)} objects: "
            + ", ".join(f"{o.class_name}({o.confidence:.2f})" for o in objects)
        )

        refined = refine_focus_map_with_objects(
            focus_map, objects, session_dir=session_dir
        )

        return refined

    except Exception as e:
        app.logger.warning(
            f"[Job {job_id}]   YOLO refinement failed (continuing without): {e}"
        )
        traceback.print_exc()
        return focus_map


# ======================== API ENDPOINTS ========================

@app.route("/ping", methods=["GET"])
def ping():
    cleanup_old_jobs()
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "yolo_available": _yolo_available,
    })


@app.route("/api/focus-stack", methods=["POST"])
def focus_stack_submit():
    try:
        files = request.files.getlist("files")
        if not files or len(files) == 0:
            return jsonify({"error": "No files uploaded"}), 400

        focus_points_raw = request.form.get("focus_points", "[]")
        focus_points = json.loads(focus_points_raw)

        threshold = int(request.form.get("threshold", 20))
        kernel_size = int(request.form.get("kernel_size", 3))
        sigma1 = float(request.form.get("sigma1", 1.0))
        sigma2 = float(request.form.get("sigma2", 2.0))
        gain = float(request.form.get("gain", 2.0))

        image_data_list = []
        filenames = []
        for f in files:
            image_data_list.append(f.read())
            filenames.append(f.filename or "unknown.jpg")

        app.logger.info(
            f"Received {len(files)} images, {len(focus_points)} focus points"
        )

        job_id = str(uuid.uuid4())[:12]

        with jobs_lock:
            jobs[job_id] = {
                "status": "queued",
                "step": "queued",
                "result_path": None,
                "error": None,
                "session_dir": None,
                "created_at": time.time(),
            }

        thread = threading.Thread(
            target=process_job,
            args=(job_id, image_data_list, focus_points,
                  threshold, kernel_size, sigma1, sigma2, gain, filenames),
            daemon=True
        )
        thread.start()

        app.logger.info(f"Job {job_id} submitted, processing in background")

        return jsonify({
            "job_id": job_id,
            "status": "queued"
        }), 202

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/<job_id>", methods=["GET"])
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job_id,
        "status": job["status"],
        "step": job.get("step", "unknown"),
    }

    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")

    return jsonify(response), 200


@app.route("/api/result/<job_id>", methods=["GET"])
def job_result(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] != "done":
        return jsonify({
            "error": "Job not ready",
            "status": job["status"]
        }), 409

    result_path = job.get("result_path")
    if not result_path or not os.path.exists(result_path):
        return jsonify({"error": "Result file not found"}), 500

    return send_file(
        result_path,
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="composite.jpg"
    )


# ======================== DETAILS MASK ========================

def generate_details_mask(
        image_path: str,
        threshold: int = 20,
        kernel_size: int = 3,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        gain: float = 2.0,
        thickness: float = 1.5
) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    dog = np.abs(g1 - g2) * gain
    if thickness > 1.0:
        dog = cv2.GaussianBlur(dog, (0, 0), thickness - 1.0)
    blurred = cv2.GaussianBlur(dog, (0, 0), 1.0)
    dog = dog + 0.7 * (dog - blurred)
    max_val = dog.max()
    if max_val > 0:
        dog = dog * (255.0 / max_val)
    dog = np.clip(dog, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


# ======================== DUPLICATE FILTER ========================

def filter_duplicate_masks(
        masks: list, similarity_threshold: float = 0.05
) -> list:
    unique = [0]
    for i in range(1, len(masks)):
        is_dup = False
        for j in unique:
            m1 = (masks[i] > 127).astype(np.uint8)
            m2 = (masks[j] > 127).astype(np.uint8)
            if m1.shape != m2.shape:
                m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]))
            xor = cv2.bitwise_xor(m1, m2)
            diff_ratio = np.count_nonzero(xor) / xor.size
            if diff_ratio <= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(i)
    return unique


# ======================== ALIGNMENT ========================

def align_images(
        image_paths: list, masks: list,
        max_features: int = 2000, max_dim: int = 1600
) -> tuple:
    ref_img = cv2.imread(image_paths[0])
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale)
    else:
        ref_small = ref_gray
    orb = cv2.ORB_create(max_features)
    ref_kp, ref_desc = orb.detectAndCompute(ref_small, None)
    aligned_images = [ref_img]
    aligned_masks = [masks[0]]
    homographies = [None]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    for i in range(1, len(image_paths)):
        src_img = cv2.imread(image_paths[i])
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            src_small = cv2.resize(src_gray, None, fx=scale, fy=scale)
        else:
            src_small = src_gray
        src_kp, src_desc = orb.detectAndCompute(src_small, None)
        H = None
        if ref_desc is not None and src_desc is not None:
            matches = bf.knnMatch(src_desc, ref_desc, k=2)
            good = []
            for m_pair in matches:
                if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
                    good.append(m_pair[0])
            if len(good) >= 10:
                inv_scale = 1.0 / scale
                src_pts = np.float32([
                    [src_kp[m.queryIdx].pt[0] * inv_scale,
                     src_kp[m.queryIdx].pt[1] * inv_scale] for m in good
                ]).reshape(-1, 1, 2)
                ref_pts = np.float32([
                    [ref_kp[m.trainIdx].pt[0] * inv_scale,
                     ref_kp[m.trainIdx].pt[1] * inv_scale] for m in good
                ]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)
        if H is not None:
            warped_img = cv2.warpPerspective(
                src_img, H, (ref_img.shape[1], ref_img.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            warped_mask = cv2.warpPerspective(
                masks[i], H, (ref_img.shape[1], ref_img.shape[0]),
                flags=cv2.INTER_NEAREST
            )
            aligned_images.append(warped_img)
            aligned_masks.append(warped_mask)
        else:
            aligned_images.append(src_img)
            aligned_masks.append(masks[i])
        homographies.append(H)
    return aligned_images, aligned_masks, homographies


# ======================== FOCUS MAP ========================

def build_focus_map(
        aligned_masks: list,
        focus_points: list,
        blur_radius: int = 31
) -> np.ndarray:
    h, w = aligned_masks[0].shape[:2]
    n = len(aligned_masks)
    float_masks = []
    for mask in aligned_masks:
        fm = mask.astype(np.float32) / 255.0
        if blur_radius > 1:
            k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
            fm = cv2.GaussianBlur(fm, (k, k), 0)
        float_masks.append(fm)
    stacked = np.stack(float_masks, axis=0)
    assignment = np.argmax(stacked, axis=0).astype(np.int32)
    max_vals = np.max(stacked, axis=0)
    unfocused = max_vals < 0.001
    if np.any(unfocused) and focus_points:
        seeds = np.zeros((h, w), dtype=np.int32)
        for i, fp in enumerate(focus_points):
            cx = int(np.clip(fp[0], 0, w - 1))
            cy = int(np.clip(fp[1], 0, h - 1))
            cv2.circle(seeds, (cx, cy), 5, i + 1, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filled = seeds.copy().astype(np.uint8)
        for _ in range(max(h, w) // 3):
            dilated = cv2.dilate(filled, kernel)
            still_zero = filled == 0
            filled[still_zero] = dilated[still_zero]
            if not np.any(filled == 0):
                break
        for y in range(h):
            for x in range(w):
                if unfocused[y, x] and filled[y, x] > 0:
                    assignment[y, x] = filled[y, x] - 1
    return assignment


# ======================== COMPOSITE ========================

def composite_images(
        aligned_images: list, focus_map: np.ndarray
) -> np.ndarray:
    h, w = focus_map.shape
    n = len(aligned_images)
    stacked = np.stack(aligned_images, axis=0)
    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    src_idx = np.clip(focus_map, 0, n - 1)
    composite = stacked[src_idx, y_idx, x_idx]
    return composite


# ======================== SEAM BLEND (FIXED) ========================

def blend_seams(
        composite: np.ndarray,
        aligned_images: list,
        focus_map: np.ndarray,
        blend_radius: int = 15
) -> np.ndarray:
    h, w = focus_map.shape
    n = len(aligned_images)

    # Find seam pixels
    padded = np.pad(focus_map, 1, mode='edge')
    seam_mask = np.zeros((h, w), dtype=np.uint8)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            seam_mask |= (focus_map != shifted).astype(np.uint8)

    # Dilate seam region
    if blend_radius > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (blend_radius * 2 + 1, blend_radius * 2 + 1)
        )
        seam_mask = cv2.dilate(seam_mask, kernel)

    seam_pixels = np.where(seam_mask > 0)
    if len(seam_pixels[0]) == 0:
        return composite

    # Build distance-based weight maps
    weight_maps = []
    for i in range(n):
        source_mask = (focus_map == i).astype(np.uint8) * 255
        dist = cv2.distanceTransform(source_mask, cv2.DIST_L2, 3)
        dist = np.minimum(dist, blend_radius).astype(np.float64)
        dist /= blend_radius
        weight_maps.append(dist)

    weight_sum = np.sum(weight_maps, axis=0) + 1e-6

    # Vectorized seam blending (much faster than per-pixel loop)
    result = composite.copy().astype(np.float64)
    sy, sx = seam_pixels

    blended = np.zeros((len(sy), 3), dtype=np.float64)
    for i in range(n):
        w_vals = (weight_maps[i][sy, sx] / weight_sum[sy, sx])[:, np.newaxis]
        blended += aligned_images[i][sy, sx].astype(np.float64) * w_vals

    result[sy, sx] = np.clip(blended, 0, 255)

    return result.astype(np.uint8)


# ======================== MAIN ========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir", default="./debug_output",
                        help="Directory for intermediate debug files")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Disable YOLO segmentation refinement")
    args = parser.parse_args()

    DEBUG_OUTPUT_DIR = args.output_dir
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    # Check YOLO availability
    if not args.no_yolo:
        try:
            from yolo_segmenter import is_available, _ensure_model
            if is_available():
                print("[INFO] Pre-loading YOLO model...")
                _ensure_model()
                _yolo_available = True
                print("[INFO] ✅ YOLO segmentation enabled")
            else:
                print("[INFO] ⚠️  ultralytics not installed, YOLO disabled")
        except Exception as e:
            print(f"[INFO] ⚠️  YOLO init failed: {e}")
            _yolo_available = False
    else:
        print("[INFO] YOLO disabled by --no-yolo flag")

    print(f"[INFO] Debug output dir: {os.path.abspath(DEBUG_OUTPUT_DIR)}")
    print(f"[INFO] YOLO refinement: {'ON' if _yolo_available else 'OFF'}")
    print(f"[INFO] API endpoints:")
    print(f"  GET  /ping                - health check")
    print(f"  POST /api/focus-stack     - submit job (returns job_id)")
    print(f"  GET  /api/status/<job_id> - poll status")
    print(f"  GET  /api/result/<job_id> - download result")

    app.run(host=args.host, port=args.port, debug=args.debug)