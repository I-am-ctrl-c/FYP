# debug_npy_to_png.py
"""
调试工具：将指定的 .npy 文件（唇部裁切图像序列）转换回一帧帧的 PNG 图像，
并在图像上绘制出目标关键点，以验证对齐和裁切效果。

说明：
- 本脚本不再需要任何运行时输入参数；
- 如需更改输入 .npy 路径或配置目录，请修改下方常量。
"""
import json
from pathlib import Path
import cv2
import numpy as np

# 这部分常量需要和 lip_cropper.py 中保持一致
TEMPLATE_5 = np.array([
    [85, 110],
    [171, 110],
    [128, 150],
    [98, 190],
    [158, 190],
], dtype=np.float32)

# 运行时内部变量（无需命令行参数）
# 相对路径以本文件所在目录为根目录
NPY_FILE: str = "data/preTrainingData/videoClip/lip_npy/20090626_section_4_000.54_004_idx0000_s0.00_t2.61_lip96.npy"
CONFIG_DIR: str = "DataPreparing"  # 包含 pipline_config.yaml 的目录

def _mouth_center() -> tuple[int, int]:
    ml = TEMPLATE_5[3]
    mr = TEMPLATE_5[4]
    return int(round((ml[0] + mr[0]) / 2.0)), int(round((ml[1] + mr[1]) / 2.0))

def main():
    project_root = Path(__file__).resolve().parent
    npy_path = project_root / NPY_FILE
    config_dir = project_root / CONFIG_DIR

    if not npy_path.exists():
        print(f"[错误] 文件不存在: {npy_path}")
        return

    # --- 1. 从 config 和 meta 文件中加载信息 ---
    try:
        # 动态加载 config 以获取输出目录
        import yaml
        config_path = config_dir / "pipline_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        output_dir_name = config.get("output_dir", "outputs")
        meta_path = project_root / output_dir_name / "lip_meta.json"
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_all = json.load(f)
            
    except FileNotFoundError as e:
        print(f"[错误] 找不到必要的配置文件: {e}")
        print("请确保您已成功运行过 pipline 并且路径设置正确。")
        return
    except Exception as e:
        print(f"[错误] 加载配置或元数据时出错: {e}")
        return

    # --- 2. 找到当前 npy 文件对应的元数据 ---
    npy_relative_path = npy_path.relative_to(project_root).as_posix()
    if npy_relative_path not in meta_all:
        print(f"[错误] 在 {meta_path} 中找不到关于 {npy_relative_path} 的元数据。")
        print("请确认 .npy 文件路径是否正确，或者它是否是一个被成功处理过的文件。")
        return
        
    meta_this_clip = meta_all[npy_relative_path]
    roi_size = meta_this_clip.get("roi_size")
    roi_wh = meta_this_clip.get("roi_wh")
    per_frame_meta = meta_this_clip.get("per_frame", [])

    if roi_wh and isinstance(roi_wh, list) and len(roi_wh) == 2:
        roi_w, roi_h = int(roi_wh[0]), int(roi_wh[1])
    else:
        if not roi_size:
            print("[错误] 元数据中缺少 'roi_size'。")
            return
        roi_w = roi_h = int(roi_size)

    # --- 3. 加载 npy 数据并准备输出目录 ---
    frames_array = np.load(npy_path)
    
    output_dir = project_root / "debug_output" / "npy_to_png" / npy_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[信息] .npy 文件包含 {len(frames_array)} 帧。")
    print(f"[信息] PNG 图像将保存到: {output_dir}")

    # --- 4. 计算模板关键点在最终裁切图中的坐标 ---
    cx_t, cy_t = _mouth_center()
    half_w = int(roi_w) // 2
    half_h = int(roi_h) // 2
    crop_origin_x = cx_t - half_w
    crop_origin_y = cy_t - half_h
    template_in_crop_coords = TEMPLATE_5 - np.array([crop_origin_x, crop_origin_y])

    # --- 5. 遍历每一帧，绘制关键点并保存 ---
    for i, frame in enumerate(frames_array):
        # BGR to RGB for display if needed, but cv2 saves BGR
        vis_frame = frame.copy()
        
        frame_meta = per_frame_meta[i] if i < len(per_frame_meta) else {}
        is_ok = frame_meta.get("ok", False)
        
        # 根据对齐是否成功，选择不同颜色的点
        # 绿色: 对齐成功, 红色: 对齐失败（使用了默认矩阵）
        dot_color = (0, 255, 0) if is_ok else (0, 0, 255)

        # 在裁切后的图像上绘制标准模板的关键点
        for point in template_in_crop_coords:
            x, y = int(round(point[0])), int(round(point[1]))
            # 确保点在图像范围内
            if 0 <= x < roi_w and 0 <= y < roi_h:
                cv2.circle(vis_frame, (x, y), 1, dot_color, -1)
        
        out_path = output_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(out_path), vis_frame)

    print(f"[完成] {len(frames_array)} 帧图像已成功保存。")

if __name__ == "__main__":
    main()
