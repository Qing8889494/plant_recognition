import xml.etree.ElementTree as ET
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
BASE_DIR = Path(r"D:\作业\intelligent_agriculture\tomato")
IMAGE_DIR = BASE_DIR / "images"
LABEL_DIR = BASE_DIR / "labels"

# 三个子集：train / val / test
SPLITS = ["train", "val", "test"]


def collect_classes(label_dirs: list[Path]) -> list[str]:
    """遍历所有 XML，收集出现的类别名（按字母排序）。"""
    classes_set = set()
    for label_dir in label_dirs:
        for xml_path in label_dir.glob("*.xml"):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                classes_set.add(name)
    return sorted(classes_set)


def convert_xml_to_yolo(xml_path: Path, class_to_id: dict[str, int]) -> list[str]:
    """
    解析单个 Pascal VOC XML，返回 YOLO 格式行列表。
    每行：class_id x_center y_center width height  （归一化到 0~1）
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in class_to_id:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 转为 YOLO 归一化坐标
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        cls_id = class_to_id[cls_name]
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def main():
    # 收集类别
    label_dirs = [LABEL_DIR / s for s in SPLITS]
    classes = collect_classes(label_dirs)
    class_to_id = {name: i for i, name in enumerate(classes)}

    print(f"Found {len(classes)} classes:")
    for name, idx in class_to_id.items():
        print(f"  {idx} -> {name}")

    # 写出 classes.txt（放在 labels 根目录供 YOLO 训练读取）
    classes_path = BASE_DIR / "classes.txt"
    classes_path.write_text("\n".join(classes), encoding="utf-8")
    print(f"\nSaved classes to: {classes_path}")

    # 逐个子集转换
    for split in SPLITS:
        xml_dir = LABEL_DIR / split

        xml_files = sorted(xml_dir.glob("*.xml"))
        if not xml_files:
            print(f"\n[WARN] {split}: no XML files, skip")
            continue

        for xml_path in xml_files:
            try:
                yolo_lines = convert_xml_to_yolo(xml_path, class_to_id)
            except Exception as e:
                print(f"  [FAIL] {xml_path.name}: {e}")
                continue

            txt_path = xml_dir / f"{xml_path.stem}.txt"
            txt_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        print(f"\n[OK] {split}: converted {len(xml_files)} files")

    print("\nDone!")


if __name__ == "__main__":
    main()
