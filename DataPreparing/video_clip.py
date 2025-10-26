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
import ffmpeg
import utils
import json
import os
import cv2
import numpy as np
import concurrent.futures
from concurrent.futures import as_completed

# -----------------------------
# 通用：配置 & 工具
# -----------------------------
def load_config(filename: str = "pipline_config.yaml") -> dict:
    cfg_path = Path(__file__).resolve().parent / filename
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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
# 输入：train.csv 中的一条原始视频名字符串（如 s11/...）
# 输出：该视频对应的切片记录列表 & 掩码字典（仅包含该视频生成的条目）
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
        # 原视频绝对路径
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

# 后端优先 face_alignment，其次 mediapipe（至少安装一个；只跑切片可不装）
USE_FACE_ALIGNMENT = True
try:
    import face_alignment  # type: ignore
except Exception:
    USE_FACE_ALIGNMENT = False

try:
    import mediapipe as mp  # type: ignore
    _MP_OK = True
except Exception:
    _MP_OK = False

DLIB_68 = {
    "left_eye_outer": 36,
    "right_eye_outer": 45,
    "nose_tip": 30,
    "mouth_left": 48,
    "mouth_right": 54,
    "mouth_all": list(range(48, 68)),
}
CANVAS = 256
TEMPLATE_5 = np.array([
    [85, 110],
    [171, 110],
    [128, 150],
    [98, 190],
    [158, 190],
], dtype=np.float32)

def _mouth_center() -> Tuple[int, int]:
    ml = TEMPLATE_5[3]; mr = TEMPLATE_5[4]
    return int(round((ml[0]+mr[0])/2.0)), int(round((ml[1]+mr[1])/2.0))

def _ema_np(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None: return cur.copy()
    return prev*(1.0-alpha) + cur*alpha

class LandmarkDetector:
    """在子进程内初始化，避免 pickling 错误"""
    def __init__(self):
        self.backend = None
        self._fa = None
        self._mp_ctx = None
        if USE_FACE_ALIGNMENT:
            try:
                import face_alignment as fa
                self._fa = fa.FaceAlignment(fa.LandmarksType._2D, flip_input=False, device='cpu')
                self.backend = "face_alignment"
            except Exception:
                self._fa = None
        if self._fa is None and _MP_OK:
            import mediapipe as mp
            self._mp_ctx = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            self.backend = "mediapipe"
        if self.backend is None:
            raise RuntimeError("唇ROI阶段需要安装 face-alignment 或 mediapipe 至少一个。")

    def detect_5pts(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """返回 5 点（两眼外角、鼻尖、两嘴角）与 conf；失败返回 (None,0.0)"""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "face_alignment":
            preds = self._fa.get_landmarks(rgb)
            if preds is None or len(preds) == 0:
                return None, 0.0
            lms = preds[0].astype(np.float32)
            five = np.stack([
                lms[DLIB_68["left_eye_outer"]],
                lms[DLIB_68["right_eye_outer"]],
                lms[DLIB_68["nose_tip"]],
                lms[DLIB_68["mouth_left"]],
                lms[DLIB_68["mouth_right"]],
            ], axis=0).astype(np.float32)
            return five, 1.0
        # mediapipe
        res = self._mp_ctx.process(rgb)
        if not res.multi_face_landmarks:
            return None, 0.0
        fl = res.multi_face_landmarks[0].landmark
        # 眼角33/263，鼻尖1，嘴角61/291
        idxs = [33,263,1,61,291]
        five = np.array([[fl[i].x*w, fl[i].y*h] for i in idxs], dtype=np.float32)
        confs = [getattr(fl[i], "visibility", 0.5) for i in idxs]
        return five, float(np.mean(confs))

def _worker_lip_from_clip(args) -> Tuple[Optional[Dict], Optional[Tuple[str, Dict]], str]:
    """
    子进程：对单个 clip 做五点对齐→裁唇→保存 .npy
    返回：(index_row, (rel_out, meta), status)
    """
    (
        clip_path_str,
        repo_root_str,
        out_dir_lip_npy_str,
        roi_size,
        ema_alpha_mat,
        min_conf,
        keep_meta,
    ) = args

    clip_path = Path(clip_path_str)
    repo_root = Path(repo_root_str)
    out_dir = Path(out_dir_lip_npy_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        det = LandmarkDetector()
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return None, None, f"[err] 无法打开切片：{clip_path}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cx_t, cy_t = _mouth_center()
        half = roi_size // 2
        ema_mat: Optional[np.ndarray] = None
        frames = []
        meta = {"frames":0, "fps":float(fps), "roi_size":int(roi_size), "method":det.backend}
        if keep_meta: meta["per_frame"] = []

        while True:
            ret, frame = cap.read()
            if not ret: break
            five, conf = det.detect_5pts(frame)
            if five is None or conf < float(min_conf):
                M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
            else:
                M_est, _ = cv2.estimateAffinePartial2D(five, TEMPLATE_5, method=cv2.LMEDS)
                if M_est is None:
                    M_est = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
                M = _ema_np(ema_mat, M_est, float(ema_alpha_mat))
            ema_mat = M.copy()

            aligned = cv2.warpAffine(frame, M, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            x1 = max(0, cx_t - half); y1 = max(0, cy_t - half)
            x2 = min(CANVAS, cx_t + half); y2 = min(CANVAS, cy_t + half)
            crop = aligned[y1:y2, x1:x2]
            if crop.shape[:2] != (roi_size, roi_size):
                crop = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
            frames.append(crop)
            if keep_meta:
                meta["per_frame"].append({"ok": five is not None and conf>=float(min_conf),
                                          "conf": float(conf),
                                          "warp": M.astype(np.float32).tolist()})
            meta["frames"] += 1

        cap.release()
        if not frames:
            return None, None, f"[err] 空视频（无帧）：{clip_path}"

        arr = np.stack(frames, axis=0)  # [T,H,W,C]
        out_name = f"{clip_path.stem}_lip{roi_size}.npy"
        out_path = out_dir / out_name
        np.save(out_path, arr)

        rel_clip = clip_path.relative_to(repo_root).as_posix()
        rel_out = out_path.relative_to(repo_root).as_posix()
        index_row = {"clip_path": rel_clip, "roi_path": rel_out, "frames": meta["frames"], "fps": meta["fps"], "roi_size": roi_size}
        return index_row, (rel_out, meta if keep_meta else {}), "OK"

    except Exception as e:
        return None, None, f"[err] 唇ROI失败：{clip_path} | {e}"

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

    # 输出 CSV / JSON
    csv_path = OUTPUT_DIR / "clipvideo.csv"
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
    EMA_ALPHA_MAT = float(config.get("lip_ema_alpha_mat", 0.35))
    MIN_CONF = float(config.get("lip_min_conf", 0.2))
    KEEP_META = bool(config.get("lip_keep_meta", True))

    clip_paths = sorted(video_dir.glob("*.mp4"))
    if not clip_paths:
        print(f"[warn] Stage-2：未找到切片：{video_dir} 下没有 .mp4")
        return

    env_workers = int(os.environ.get("LIP_NPY_WORKERS", "0") or "0")
    num_workers = env_workers if env_workers > 0 else max(1, (os.cpu_count() or 4) - 1)

    jobs = [
        (str(p), str(REPO_ROOT), str(out_dir_lip_npy), ROI_SIZE, EMA_ALPHA_MAT, MIN_CONF, KEEP_META)
        for p in clip_paths
    ]
    lip_index_csv = OUTPUT_DIR / "lip_index.csv"
    lip_meta_json = OUTPUT_DIR / "lip_meta.json"
    index_rows: List[Dict] = []
    meta_all: Dict[str, Dict] = {}

    print(f"[info] Stage-2 唇ROI：{len(jobs)} 个切片，并行启动（workers={num_workers}）…")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_worker_lip_from_clip, job) for job in jobs]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            index_row, meta_tuple, status = fut.result()
            if status == "OK" and index_row:
                index_rows.append(index_row)
                if KEEP_META and meta_tuple:
                    rel_out, meta = meta_tuple
                    meta_all[rel_out] = meta
            else:
                print(status)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[stage-2] 进度：{done}/{total}")

    with open(lip_index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "roi_path", "frames", "fps", "roi_size"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"[info] 写出：{lip_index_csv}（{len(index_rows)} 条）")

    if KEEP_META:
        with open(lip_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta_all, f, ensure_ascii=False)
        print(f"[info] 写出：{lip_meta_json}（keys={len(meta_all)}）")

# -----------------------------
# main
# -----------------------------
def main():
    config = load_config()
    # Stage-1：并行切片
    run_stage1_slice_parallel(config)

    # Stage-2：是否开启唇ROI（可在 config 里开关）
    if bool(config.get("enable_lip_stage", True)):
        run_stage2_lip_parallel(config)
    else:
        print("[info] 已跳过 Stage-2（enable_lip_stage = false）")

if __name__ == "__main__":
    main()
