# lip_cropper.py
"""
从视频文件中提取、对齐并裁切唇部 ROI（稳健版：相似变换 + 失败回退 + 参数空间EMA）。
"""
from typing import Optional, Tuple, List
import os
from pathlib import Path
import numpy as np
import cv2

# 限制并行库默认线程数，避免多进程×多线程导致资源竞争/卡顿（可被外部覆盖）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# 后端优先 face_alignment，其次 mediapipe
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

# 对齐画布与五点模板（像素坐标，需同一坐标系）
CANVAS = 256
TEMPLATE_5 = np.array([
    [85, 110],   # 左眼外角
    [171, 110],  # 右眼外角
    [128, 150],  # 鼻尖
    [98, 190],   # 左嘴角
    [158, 190],  # 右嘴角
], dtype=np.float32)

def _mouth_center() -> Tuple[int, int]:
    ml = TEMPLATE_5[3]; mr = TEMPLATE_5[4]
    return int(round((ml[0]+mr[0])/2.0)), int(round((ml[1]+mr[1])/2.0))

# ------- 相似变换工具：Umeyama + 参数分解/合成 + EMA -------

def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Umeyama (1991) 闭式解：估计相似变换（旋转+等比缩放+平移）
    src, dst: (N,2) float32/64
    返回: 2x3 float32 矩阵
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    assert src.shape == dst.shape and src.shape[1] == 2

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]  # 2x2
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    # 反射处理
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt

    var_src = (src_c ** 2).sum() / src.shape[0]
    scale = S.sum() / max(var_src, 1e-12)
    t = mu_dst - scale * (R @ mu_src)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = (scale * R).astype(np.float32)
    M[:, 2] = t.astype(np.float32)
    return M

def decompose_similarity(M: np.ndarray) -> Tuple[float, float, float, float]:
    """从 2x3 相似矩阵提取 (scale, theta, tx, ty)"""
    a, b = float(M[0, 0]), float(M[0, 1])
    scale = float(np.hypot(a, b))
    theta = float(np.arctan2(b, a))
    tx, ty = float(M[0, 2]), float(M[1, 2])
    return scale, theta, tx, ty

def compose_similarity(scale: float, theta: float, tx: float, ty: float) -> np.ndarray:
    """由 (scale, theta, tx, ty) 合成 2x3 相似矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    M = np.array([[scale * c, scale * s, tx],
                  [-scale * s, scale * c, ty]], dtype=np.float32)
    return M

def ema_params(prev: Optional[Tuple[float, float, float, float]],
               curr: Tuple[float, float, float, float],
               alpha: float) -> Tuple[float, float, float, float]:
    """在参数空间做 EMA：prev*(1-alpha) + curr*alpha"""
    if prev is None:
        return curr
    p = np.array(prev, dtype=np.float32)
    c = np.array(curr, dtype=np.float32)
    out = (1.0 - alpha) * p + alpha * c
    return tuple(map(float, out))

# 可选：如 detector 五点顺序与 TEMPLATE_5 不一致，在此做重排
def reorder_landmarks_if_needed(pts: np.ndarray) -> np.ndarray:
    """
    目标顺序： [左眼外角, 右眼外角, 鼻尖, 左嘴角, 右嘴角]
    你的 detector 如已满足，直接返回原值；否则在此重映射。
    """
    return pts

# ------- 关键点检测 -------

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

    def close(self) -> None:
        """释放底层资源，避免在子进程中残留线程阻塞退出（特别是 MediaPipe）。"""
        try:
            if self.backend == "mediapipe" and self._mp_ctx is not None:
                try:
                    # MediaPipe Solution 支持显式关闭
                    self._mp_ctx.close()
                except Exception:
                    pass
                finally:
                    self._mp_ctx = None
        except Exception:
            pass

    def detect_5pts(self, frame_bgr: np.ndarray, output_path: Optional[str] = None) -> Tuple[
        Optional[np.ndarray], float]:
        """返回 5 点（两眼外角、鼻尖、两嘴角）与 conf；失败返回 (None,0.0)。无论成功与否，若给了 output_path，就落一张调试图。"""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        five: Optional[np.ndarray] = None
        conf: float = 0.0

        if self.backend == "face_alignment":
            preds = self._fa.get_landmarks(rgb)
            if preds is not None and len(preds) > 0:
                lms = preds[0].astype(np.float32)
                five = np.stack([
                    lms[DLIB_68["left_eye_outer"]],
                    lms[DLIB_68["right_eye_outer"]],
                    lms[DLIB_68["nose_tip"]],
                    lms[DLIB_68["mouth_left"]],
                    lms[DLIB_68["mouth_right"]],
                ], axis=0).astype(np.float32)
                conf = 1.0
        else:  # mediapipe
            res = self._mp_ctx.process(rgb)
            if res.multi_face_landmarks:
                fl = res.multi_face_landmarks[0].landmark
                idxs = [33, 263, 1, 61, 291]
                five = np.array([[fl[i].x * w, fl[i].y * h] for i in idxs], dtype=np.float32)
                confs = [getattr(fl[i], "visibility", 0.5) for i in idxs]
                conf = float(np.mean(confs))

        # —— 调试图：无论成功与否都落盘 ——
        if output_path:
            dbg = frame_bgr.copy()
            if five is not None:
                for x, y in five:
                    cv2.circle(dbg, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
                cv2.putText(dbg, f"backend={self.backend} conf={conf:.3f}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(dbg, f"NO LANDMARKS (backend={self.backend})", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, dbg)

        if five is None:
            return None, 0.0
        return five, conf


# ------- 进程内单例缓存，避免每个任务重复初始化后端 -------
_CACHED_DETECTOR: Optional[LandmarkDetector] = None

def _get_or_create_detector() -> LandmarkDetector:
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is None:
        _CACHED_DETECTOR = LandmarkDetector()
    return _CACHED_DETECTOR

def _release_cached_detector() -> None:
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is not None:
        try:
            _CACHED_DETECTOR.close()
        except Exception:
            pass
        _CACHED_DETECTOR = None

# 供多进程初始化器调用：在 worker 启动时预热 Detector
def init_detector_for_worker() -> None:
    _get_or_create_detector()

# 供需要时显式释放（通常不必，进程退出会回收）
def release_detector_for_worker() -> None:
    _release_cached_detector()

# ------- 主流程：相似变换 + 回退 + 参数EMA -------

def crop_lip_from_video_file(
    video_path: str,
    roi_size: int,
    ema_alpha: float,
    debug_output_dir: Optional[str] = None,
    roi_wh: Optional[Tuple[int, int]] = None,
    pause_each_frame: bool = False,   # 新增调试开关（默认连续播放）
    use_cached_detector: bool = True, # 进程内共享 Detector，避免频繁构造
) -> Tuple[Optional[np.ndarray], dict]:
    """
    从单个视频文件中裁切唇部 ROI 序列（稳健版）：
    - 估计相似变换（Umeyama），避免剪切/拉伸；
    - 检测失败时回退上一帧变换，而不是单位矩阵；
    - 在 (scale, theta, tx, ty) 参数空间进行 EMA 平滑。
    """
    try:
        # 1) 初始化（默认复用进程内单例，避免频繁初始化导致的卡顿/资源竞争）
        det = _get_or_create_detector() if use_cached_detector else LandmarkDetector()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        cx_t, cy_t = _mouth_center()
        if roi_wh is not None and len(roi_wh) == 2:
            roi_w, roi_h = int(roi_wh[0]), int(roi_wh[1])
        else:
            roi_w = roi_h = int(roi_size)
        half_w, half_h = roi_w // 2, roi_h // 2

        # EMA 状态：上一帧的 (s,theta,tx,ty)
        last_params: Optional[Tuple[float, float, float, float]] = None
        frames: List[np.ndarray] = []

        meta = {
            "frames": 0,
            "fps": float(fps),
            "roi_size": int(roi_size),
            "roi_wh": [roi_w, roi_h],
            "method": det.backend,
            "per_frame": [],
        }

        dbg_dir = Path(debug_output_dir) if debug_output_dir else None
        if dbg_dir is not None:
            dbg_dir.mkdir(parents=True, exist_ok=True)
            video_name = Path(video_path).stem

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 2) 检测并估计当前相似变换
            five, conf = det.detect_5pts(frame)
            ok = False
            params_now: Optional[Tuple[float, float, float, float]] = None
            if five is not None:
                five = reorder_landmarks_if_needed(np.asarray(five, dtype=np.float32))
                M_now = umeyama_similarity(five, TEMPLATE_5.astype(np.float32))
                params_now = decompose_similarity(M_now)
                ok = True

            # 3) 失败回退/EMA 平滑
            if not ok and last_params is None:
                # 第一帧就失败，无法定位，直接跳过该帧
                meta["per_frame"].append({"ok": False, "conf": float(conf),
                                          "warp": np.eye(2, 3, dtype=np.float32).tolist()})
                idx += 1
                continue

            if ok:
                smoothed = ema_params(last_params, params_now, ema_alpha)
            else:
                smoothed = last_params  # 沿用上一帧

            M_use = compose_similarity(*smoothed)
            last_params = smoothed

            # 4) 对齐与裁剪
            aligned = cv2.warpAffine(
                frame, M_use, (CANVAS, CANVAS),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

            x1 = max(0, cx_t - half_w); y1 = max(0, cy_t - half_h)
            x2 = min(CANVAS, cx_t + half_w); y2 = min(CANVAS, cy_t + half_h)
            crop = aligned[y1:y2, x1:x2]
            if crop.shape[1] != roi_w or crop.shape[0] != roi_h:
                crop = cv2.resize(crop, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

            # 5) 调试输出
            if dbg_dir is not None:
                dbg = aligned.copy()
                # 模板点（蓝）
                for p in TEMPLATE_5:
                    px, py = int(round(float(p[0]))), int(round(float(p[1])))
                    cv2.circle(dbg, (px, py), 2, (255, 0, 0), -1)
                # 变换后的检测点（红）
                if five is not None:
                    five_h = np.hstack([five.astype(np.float32), np.ones((5, 1), dtype=np.float32)])
                    five_warp = (M_use @ five_h.T).T
                    for q in five_warp:
                        cv2.circle(dbg, (int(round(q[0])), int(round(q[1]))), 2, (0, 0, 255), -1)
                # 裁剪框（黄）
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 1)

                # 在多进程环境下避免使用 HighGUI 窗口，防止卡死；仅在逐帧暂停调试时显示。
                if pause_each_frame:
                    try:
                        cv2.imshow("Aligned Frame", dbg)
                        cv2.waitKey(0)
                    except Exception:
                        pass

                cv2.imwrite(str(dbg_dir / f"{video_name}_aligned_{idx:04d}.png"), dbg)

            # 6) 记录 & 收集
            frames.append(crop)
            meta["per_frame"].append({"ok": bool(ok), "conf": float(conf), "warp": M_use.astype(np.float32).tolist()})
            meta["frames"] += 1
            idx += 1

        # 7) 结束
        cap.release()
        # 若未使用缓存，则为该任务独占实例，安全关闭；否则交由进程退出或外部显式释放
        if not use_cached_detector:
            try:
                det.close()
            except Exception:
                pass
        # 仅在调试显示时销毁窗口，避免在无窗口环境/子进程中卡住
        if pause_each_frame:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if not frames:
            return None, meta

        arr = np.stack(frames, axis=0)  # [T,H,W,C]
        return arr, meta

    except Exception as e:
        print(f"[error] crop_lip_from_video_file failed for {video_path}: {e}")
        return None, {}
