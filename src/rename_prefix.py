import os
from pathlib import Path

BASE_DIR = Path(r"D:\作业\intelligent_agriculture\tomato")
PREFIX = "Tomato_"

# 需要重命名的目录：images 和 labels 下的 train / val / test
for folder in ["images", "labels"]:
    for split in ["train", "val", "test"]:
        sub_dir = BASE_DIR / folder / split
        if not sub_dir.exists():
            continue

        renamed = 0
        for file_path in sorted(sub_dir.iterdir()):
            if not file_path.is_file():
                continue

            # 只处理 jpg 和 txt 文件
            suffix = file_path.suffix.lower()
            if suffix not in (".jpg", ".txt", ".png", ".jpeg"):
                continue

            # 已有前缀则跳过，避免重复重命名
            if file_path.name.startswith(PREFIX):
                continue

            new_name = PREFIX + file_path.name
            new_path = sub_dir / new_name
            file_path.rename(new_path)
            renamed += 1

        print(f"[{folder}/{split}] renamed {renamed} files")

print("\nDone!")
