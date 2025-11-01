# This file makes the directory a Python package
from pathlib import Path

import yaml

import lip_cropper
import video_clip

def load_config(filename: str = "pipline_config.yaml") -> dict:
    cfg_path = Path(__file__).resolve().parent / filename
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    # Stage-1：并行切片
    if bool(config.get("enable_slice", True)):
        video_clip.run_stage1_slice_parallel(config)
    else:
        print("[info] 已跳过 Stage-1（enable_slice = false）")
    # Stage-2：是否开启唇ROI（可在 config 里开关）
    if bool(config.get("enable_lip_stage", True)):
        video_clip.run_stage2_lip_parallel(config)
    else:
        print("[info] 已跳过 Stage-2（enable_lip_stage = false）")

if __name__ == "__main__":
    main()