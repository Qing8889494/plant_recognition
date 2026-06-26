import os

# 替换 labels/train、val、test 中所有文件的 0 -> 1
base_dir = r"D:\\作业\\intelligent_agriculture\\your_dataset\\your_dataset\\labels"

for split in ['train', 'val', 'test']:
    label_dir = os.path.join(base_dir, split)
    if not os.path.exists(label_dir):
        continue
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            filepath = os.path.join(label_dir, file)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    # 将第一个数字改为 '1'
                    parts[0] = '1'
                    new_lines.append(' '.join(parts))
            with open(filepath, 'w') as f:
                f.write('\n'.join(new_lines))
print("标签已从 0 改为 1")