"""
最简调试脚本：不带命令行参数。
按内部变量设置的视频路径，逐帧运行 LandmarkDetector.detect_5pts，
并将标注了 5 个点的 PNG 保存到指定目录。
"""

from pathlib import Path
import cv2
import numpy as np

from DataPreparing.lip_cropper import LandmarkDetector


# ========= 内部可改变量 =========
# 输入视频路径（请修改为你本地的视频路径）
VIDEO_PATH = "data/preTrainingData/videoClip/video/20090626_section_4_000.54_004_idx0000_s0.00_t2.61.mp4"

# 输出目录（逐帧 PNG 将保存到此处）
OUTPUT_DIR = "debug_output/points_simple"

# 处理每 N 帧（默认 1 表示每一帧）
STEP = 1

# 限制最多处理多少帧（0 表示处理到视频结束）
LIMIT = 0
# ========= 内部可改变量 =========


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

    det = LandmarkDetector()
    print(f"使用后端: {det.backend}")

    processed = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if STEP > 1 and (idx % STEP != 0):
            idx += 1
            continue

        out_path = out_dir / f"frame_{idx:06d}.png"
        five, conf = det.detect_5pts(frame, output_path=str(out_path))

        if five is None:
            # detect_5pts 仅在检测成功时才会写入 output_path
            # 这里为了可见性，也保存一张原始帧并写上提示
            vis = frame.copy()
            cv2.putText(
                vis,
                "no landmarks",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(out_path), vis)
            print(f"[{idx}] 未检测到人脸关键点 -> {out_path}")
        else:
            print(f"[{idx}] conf={conf:.2f} -> {out_path}")

        processed += 1
        idx += 1
        if LIMIT > 0 and processed >= LIMIT:
            break

    cap.release()
    print(f"完成。共处理 {processed} 帧，输出目录: {out_dir}")


if __name__ == "__main__":
    main()

