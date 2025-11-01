# video_clip.py
"""
并行切片 + （可选）并行唇部ROI裁切（仿射对齐，.npy输出）
Stage-1: 原视频 -> 随机长度片段（1.5s~3.5s，或由 config 决定）
Stage-2: 片段 -> 五点对齐 -> 固定嘴唇ROI -> .npy
产物：
  OUTPUT_DIR/video/*.mp4
  OUTPUT_DIR/clipvideo.csv
  OUTPUT_DIR/clip_masks.json
  （可选）OUTPUT_DIR/lip_npy/*.npy
  （可选）OUTPUT_DIR/lip_index.csv
  （可选）OUTPUT_DIR/lip_meta.json
"""

from typing import Iterator, Optional, Dict, List, Tuple
from pathlib import Path
import hashlib
import random
import csv
import yaml
import sys
import ffmpeg
import json
import os
import cv2
import numpy as np
import concurrent.futures
import multiprocessing as mp
from concurrent.futures import as_completed
from lip_cropper import crop_lip_from_video_file
import lip_cropper

# Ensure repository root is importable so sibling modules like `utils` resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils

# -----------------------------
# 通用：配置 & 工具
# -----------------------------


def get_video_time(video_path: str) -> float:
    try:
        duration_str = ffmpeg.probe(video_path)["format"]["duration"]
        return float(duration_str)
    except Exception as e:
        print(f"[warn] 获取时长失败：{video_path} | {e}")
        return 0.0

def get_video_list() -> List[str]:
    names: List[str] = []
    with open(utils.get_train_path_str(), newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0]:
                names.append(row[0].strip())
    return names

def _stable_int_from_str(s: str, bits: int = 32) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[: bits // 4], 16)

# -----------------------------
# Stage-1：随机滑窗定义
# -----------------------------
def iter_stable_random_windows(
    total_sec: float,
    base_step_sec: float,
    jitter_sec: float,
    min_len_sec: float,
    max_len_sec: float,
    seed: Optional[int] = None,
    fps: Optional[int] = None,
) -> Iterator[Dict]:
    if total_sec <= 0:
        return
    if base_step_sec <= 0:
        raise ValueError("base_step_sec 必须 > 0")
    if min_len_sec <= 0 or max_len_sec <= 0 or max_len_sec < min_len_sec:
        raise ValueError("窗口长度范围非法")

    rng = random.Random(seed)
    start_time = 0.0
    idx = 0
    min_advance = 1e-6
    jitter_cap = min(abs(jitter_sec), base_step_sec * 0.9)

    while start_time < total_sec:
        dur = rng.uniform(min_len_sec, max_len_sec)
        valid_sec = max(0.0, min(dur, max(0.0, total_sec - start_time)))
        pad_sec = max(0.0, dur - valid_sec)

        item: Dict = {
            "index": idx,
            "start_sec": start_time,
            "dur_sec": dur,
            "valid_sec": valid_sec,
            "pad_sec": pad_sec,
        }

        if fps and fps > 0:
            target_frames = max(1, int(round(dur * fps)))
            valid_frames = max(0, int(round(valid_sec * fps)))
            pad_frames = max(0, target_frames - valid_frames)
            mask = [1] * valid_frames + [0] * pad_frames
            item.update(
                {
                    "target_frames": target_frames,
                    "valid_frames": valid_frames,
                    "mask": mask,
                }
            )

        yield item

        step_jitter = rng.uniform(-jitter_cap, jitter_cap)
        next_start = start_time + base_step_sec + step_jitter
        start_time = max(start_time + min_advance, next_start)
        idx += 1

# -----------------------------
# Stage-1：切片导出
# -----------------------------
def export_clip(
    src_path: str,
    out_path: Path,
    start_sec: float,
    dur_sec: float,
    pad_sec: float,
    target_fps: Optional[int] = None,
) -> None:
    stream = ffmpeg.input(src_path, ss=start_sec)
    if target_fps:
        stream = stream.filter("fps", fps=target_fps)
    if pad_sec > 0:
        stream = stream.filter("tpad", stop_mode="clone", stop_duration=pad_sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg.output(
            stream,
            str(out_path),
            t=dur_sec,
            vcodec="libx264",
            preset="veryfast",
            crf=23,
            acodec="aac",
            audio_bitrate="128k",
        )
        .overwrite_output()
        .run(quiet=True)
    )

# -----------------------------
# Stage-1：单视频 worker（并行）
# -----------------------------
def _worker_slice_single_video(args) -> Tuple[List[Dict], Dict[str, Dict], str]:
    (
        video_name,  # train.csv 的名字
        repo_root_str,
        output_dir_str,
        base_step_sec,
        jitter_sec,
        min_len_sec,
        max_len_sec,
        target_fps,
        global_seed,
    ) = args

    REPO_ROOT = Path(repo_root_str)
    OUTPUT_DIR = Path(output_dir_str)
    video_out_dir = OUTPUT_DIR / "video"

    records: List[Dict] = []
    masks_dict: Dict[str, Dict] = {}

    try:
        src_path = utils.get_origin_video_path_str(video_name)
        total = get_video_time(src_path)
        if total <= 0:
            return records, masks_dict, f"[skip] 视频不可读或时长为0：{src_path}"

        per_video_seed = (int(global_seed) ^ _stable_int_from_str(video_name)) & 0xFFFFFFFF

        for window in iter_stable_random_windows(
            total_sec=total,
            base_step_sec=base_step_sec,
            jitter_sec=jitter_sec,
            min_len_sec=min_len_sec,
            max_len_sec=max_len_sec,
            seed=per_video_seed,
            fps=target_fps,
        ):
            idx = window["index"]
            start_sec = window["start_sec"]
            dur_sec = window["dur_sec"]
            valid_sec = window["valid_sec"]
            pad_sec = window["pad_sec"]
            mask = window.get("mask")
            target_frames = window.get("target_frames")
            valid_frames = window.get("valid_frames")

            out_path = video_out_dir / f"{Path(video_name).stem}_idx{idx:04d}_s{start_sec:.2f}_t{dur_sec:.2f}.mp4"
            export_clip(
                src_path=src_path,
                out_path=out_path,
                start_sec=start_sec,
                dur_sec=dur_sec,
                pad_sec=pad_sec,
                target_fps=target_fps,
            )
            rel_clip_path = Path(os.path.relpath(out_path, REPO_ROOT)).as_posix()

            records.append(
                {
                    "idx": idx,
                    "clip_path": rel_clip_path,
                    "original_name": video_name,
                    "duration_sec": round(float(valid_sec), 3),
                }
            )

            masks_dict[rel_clip_path] = {
                "start_sec": round(float(start_sec), 6),
                "dur_sec": round(float(dur_sec), 6),
                "valid_sec": round(float(valid_sec), 6),
                "pad_sec": round(float(pad_sec), 6),
                "target_fps": target_fps,
                "target_frames": int(target_frames) if target_frames is not None else None,
                "valid_frames": int(valid_frames) if valid_frames is not None else None,
                "mask": mask,
            }

        return records, masks_dict, "OK"

    except Exception as e:
        return records, masks_dict, f"[err] 切片失败：{video_name} | {e}"

# =========================================================
# Stage-2（可选）：唇 ROI 对齐裁切（五点相似变换 → .npy）
# =========================================================
def _worker_lip_from_clip(args) -> Tuple[Optional[Dict], Optional[Tuple[str, Dict]], str]:
    """
    子进程：对单个 clip 做五点对齐→裁唇→保存 .npy
    (通过调用 lip_cropper.py 中的函数实现)
    返回：(index_row, (rel_out, meta), status)
    """
    (
        clip_path_str,
        repo_root_str,
        out_dir_lip_npy_str,
        roi_size,
        roi_w,
        roi_h,
        ema_alpha_mat,
        keep_meta,
        debug_output_dir,  # New
    ) = args

    clip_path = Path(clip_path_str)
    repo_root = Path(repo_root_str)
    out_dir = Path(out_dir_lip_npy_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pid = os.getpid()
        print(f"[stage-2][pid={pid}] start: {clip_path.name}")
        # 调用外部模块进行核心处理
        # 允许矩形 ROI：通过 roi_wh 传入（若与 roi_size 相等，则为方形）
        arr, meta = crop_lip_from_video_file(
            video_path=str(clip_path),
            roi_size=roi_size,
            ema_alpha=ema_alpha_mat,
            debug_output_dir=debug_output_dir,  # New
            roi_wh=(int(roi_w), int(roi_h)),
        )

        if arr is None or len(arr) == 0:
            return None, None, f"[err] 唇ROI处理返回空结果: {clip_path}"

        # 保存 .npy 文件
        if int(roi_w) == int(roi_h) == int(roi_size):
            out_name = f"{clip_path.stem}_lip{roi_size}.npy"
        else:
            out_name = f"{clip_path.stem}_lip{int(roi_w)}x{int(roi_h)}.npy"
        out_path = out_dir / out_name
        np.save(out_path, arr)
        print(
            f"[stage-2][pid={pid}] saved: {out_path.name} frames={arr.shape[0]} backend={meta.get('method','?')}"
        )

        # 准备返回的元数据
        rel_clip = clip_path.relative_to(repo_root).as_posix()
        rel_out = out_path.relative_to(repo_root).as_posix()
        
        final_meta = {
            "frames": meta.get("frames", 0),
            "fps": meta.get("fps", 25.0),
            "roi_size": meta.get("roi_size", roi_size),
            "roi_wh": meta.get("roi_wh", [int(roi_w), int(roi_h)]),
            "method": meta.get("method", "unknown"),
        }
        if keep_meta:
            final_meta["per_frame"] = meta.get("per_frame", [])

        index_row = {
            "clip_path": rel_clip,
            "roi_path": rel_out,
            "frames": final_meta["frames"],
            "fps": final_meta["fps"],
            "roi_size": roi_size,
        }
        
        meta_to_return = (rel_out, final_meta if keep_meta else {})
        
        print(f"[stage-2][pid={pid}] done: {clip_path.name}")
        return index_row, meta_to_return, "OK"

    except Exception as e:
        return None, None, f"[err] 唇ROI失败: {clip_path} | {e}"

# -----------------------------
# Orchestrator
# -----------------------------
def run_stage1_slice_parallel(config: dict) -> Tuple[List[Dict], Dict[str, Dict]]:
    REPO_ROOT: Path = utils.get_repository_path()
    OUTPUT_DIR: Path = REPO_ROOT / config["output_dir"]
    (OUTPUT_DIR / "video").mkdir(parents=True, exist_ok=True)

    BASE_STEP_SEC = float(config["base_step_sec"])
    JITTER_SEC = float(config["jitter_sec"])
    MIN_LEN_SEC = float(config["min_len_sec"])
    MAX_LEN_SEC = float(config["max_len_sec"])
    TARGET_FPS = int(config.get("target_fps", 0)) or None
    GLOBAL_SEED = int(config["global_seed"])

    list_video = get_video_list()
    print(f"[info] Stage-1 切片：{len(list_video)} 个原始视频，并行启动…")

    jobs = [
        (
            video_name,
            str(REPO_ROOT),
            str(OUTPUT_DIR),
            BASE_STEP_SEC, JITTER_SEC, MIN_LEN_SEC, MAX_LEN_SEC,
            TARGET_FPS, GLOBAL_SEED,
        )
        for video_name in list_video
    ]
    env_workers = int(os.environ.get("CLIP_WORKERS", "0") or "0")
    num_workers = env_workers if env_workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    num_workers = 8
    all_records: List[Dict] = []
    all_masks: Dict[str, Dict] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_worker_slice_single_video, job) for job in jobs]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            recs, masks, status = fut.result()
            all_records.extend(recs)
            all_masks.update(masks)
            done += 1
            if status != "OK":
                print(status)
            if done % 10 == 0 or done == total:
                print(f"[stage-1] 进度：{done}/{total}")

    csv_path = OUTPUT_DIR / "clip_video.csv"
    json_path = OUTPUT_DIR / "clip_masks.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "clip_path", "original_name", "duration_sec"])
        writer.writeheader()
        writer.writerows(all_records)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_masks, f, ensure_ascii=False)
    print(f"[info] 写出：{csv_path}（{len(all_records)} 条），{json_path}（{len(all_masks)} 键）")

    return all_records, all_masks

def run_stage2_lip_parallel(config: dict) -> None:
    """可选的 Stage-2：对 OUTPUT_DIR/video/*.mp4 并行做唇ROI→.npy"""
    REPO_ROOT: Path = utils.get_repository_path()
    OUTPUT_DIR: Path = REPO_ROOT / config["output_dir"]
    video_dir = OUTPUT_DIR / "video"
    out_dir_lip_npy = OUTPUT_DIR / "lip_npy"
    out_dir_lip_npy.mkdir(parents=True, exist_ok=True)

    ROI_SIZE = int(config.get("lip_roi_size", 96))
    ROI_W = int(config.get("lip_roi_width", 0))
    ROI_H = int(config.get("lip_roi_height", 0))
    if ROI_W <= 0 or ROI_H <= 0:
        ROI_W, ROI_H = ROI_SIZE, ROI_SIZE
    EMA_ALPHA_MAT = float(config.get("lip_ema_alpha_mat", 0.35))
    KEEP_META = bool(config.get("lip_keep_meta", True))
    
    DEBUG_LIP_DETECTION = bool(config.get("debug_lip_detection", False))
    DEBUG_OUTPUT_DIR = REPO_ROOT / "debug_output/lip_detection" if DEBUG_LIP_DETECTION else None
    if DEBUG_OUTPUT_DIR:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[debug] 唇部检测调试模式已开启，图像将保存至: {DEBUG_OUTPUT_DIR}")

    clip_paths = sorted(video_dir.glob("*.mp4"))
    if not clip_paths:
        print(f"[warn] Stage-2：未找到切片：{video_dir} 下没有 .mp4")
        return
        
    if DEBUG_LIP_DETECTION:
        print(f"[debug] 将只处理第一个视频切片进行调试: {clip_paths[0]}")
        clip_paths = clip_paths[:1]

    env_workers = int(os.environ.get("LIP_NPY_WORKERS", "0") or "0")
    num_workers = env_workers if env_workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    num_workers = 8
    jobs = [
        (
            str(p),
            str(REPO_ROOT),
            str(out_dir_lip_npy),
            ROI_SIZE,
            ROI_W,
            ROI_H,
            EMA_ALPHA_MAT,
            KEEP_META,
            str(DEBUG_OUTPUT_DIR) if DEBUG_OUTPUT_DIR else None,
        )
        for p in clip_paths
    ]
    lip_index_csv = OUTPUT_DIR / "lip_index.csv"
    lip_meta_json = OUTPUT_DIR / "lip_meta.json"
    index_rows: List[Dict] = []
    meta_all: Dict[str, Dict] = {}

    print(f"[info] Stage-2 唇ROI：{len(jobs)} 个切片，并行启动（workers={num_workers}）…")

    # 通过 initializer 让每个 worker 进程预先构造并缓存一次检测器，避免按任务反复初始化
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=lip_cropper.init_detector_for_worker,
        mp_context=mp.get_context("spawn"),
    ) as ex:
        # 采用“限流提交”策略，避免一次性提交几十万任务导致父进程/管道阻塞
        job_iter = iter(jobs)
        inflight_limit = max(8, num_workers * 4)
        future_to_clip: Dict[concurrent.futures.Future, str] = {}
        total = len(jobs)
        done = 0

        def submit_until_full():
            nonlocal future_to_clip
            while len(future_to_clip) < inflight_limit:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                fut = ex.submit(_worker_lip_from_clip, job)
                future_to_clip[fut] = Path(job[0]).stem

        submit_until_full()

        while future_to_clip:
            # 等待至少一个完成，再继续补充
            done_set, _ = concurrent.futures.wait(
                future_to_clip.keys(), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done_set:
                clip_name = future_to_clip.pop(fut)
                try:
                    index_row, meta_tuple, status = fut.result()
                except Exception as exc:
                    print(f"[stage-2][{clip_name}] 子进程异常：{exc}")
                    status = "EXCEPTION"
                    index_row = None
                    meta_tuple = None

                if status == "OK" and index_row:
                    index_rows.append(index_row)
                    if KEEP_META and meta_tuple:
                        rel_out, meta = meta_tuple
                        meta_all[rel_out] = meta
                else:
                    print(f"[stage-2][{clip_name}] {status}")

                done += 1
                if done % 10 == 0 or done == total:
                    print(f"--------- [stage-2] 进度：{done}/{total}（当前完成：{clip_name}）")

            submit_until_full()

    with open(lip_index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "roi_path", "frames", "fps", "roi_size"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"[info] 写出：{lip_index_csv}（{len(index_rows)} 条）")

    if KEEP_META:
        with open(lip_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta_all, f, ensure_ascii=False)
        print(f"[info] 写出：{lip_meta_json}（keys={len(meta_all)}）")
