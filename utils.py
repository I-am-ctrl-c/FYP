from pathlib import Path

def get_origin_video_path(name) -> Path:
    repo_dir = get_repository_path()
    folder, rest = name.split("/", 1)
    time_folder, video_name = rest.split("_", 1)

    video_name += ".mp4"
    return repo_dir / "data" / "CMLRdataset" / "video" / folder / time_folder / video_name

def get_origin_video_path_str(name) -> str:
    return str(get_origin_video_path(name))

def get_repository_path() -> Path:
    return Path(__file__).resolve().parent

def get_repository_path_str() -> str:
    return str(get_repository_path())

TRAIN_PATH: Path = get_repository_path() / "data" / "CMLRdataset" / "train.csv"
TEST_PATH: Path = get_repository_path() / "data" / "CMLRdataset" / "test.csv"
VAL_PATH: Path = get_repository_path() / "data" / "CMLRdataset" / "val.csv"

def get_train_path() -> Path:
    return TRAIN_PATH

def get_test_path() -> Path:
    return TEST_PATH

def get_val_path() -> Path:
    return VAL_PATH

def get_train_path_str() -> str:
    return str(TRAIN_PATH)

def get_test_path_str() -> str:
    return str(TEST_PATH)

def get_val_path_str() -> str:
    return str(VAL_PATH)

if __name__ == "__main__":
    print(get_repository_path())
    print(get_repository_path_str())
    print(get_origin_video_path("s11/20180203_section_1_020.26_022.14"))
    print(get_origin_video_path_str("s11/20180203_section_1_020.26_022.14"))