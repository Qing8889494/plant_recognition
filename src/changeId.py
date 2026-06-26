import os

# ！！！改成你电脑上数据集根目录的实际路径！！！
base_dir = r"D:\\作业\\intelligent_agriculture\\your_dataset"   # 例如 "C:/Users/你的用户名/Desktop/fruits_dataset"

for split in ['train', 'val', 'test']:
    label_dir = os.path.join(base_dir, split, 'labels')
    if not os.path.exists(label_dir):
        print(f"⚠️ 警告：目录不存在 {label_dir}，已跳过")
        continue

    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            filepath = os.path.join(label_dir, file)
            with open(filepath, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:          # 保留空行
                    new_lines.append('')
                    continue
                parts = stripped.split()
                if parts:
                    # 只把原本类别ID为 '0' 的改成 '1'
                    if parts[0] == '0':
                        parts[0] = '1'
                    new_lines.append(' '.join(parts))
                else:
                    new_lines.append(stripped)

            with open(filepath, 'w') as f:
                f.write('\n'.join(new_lines))

print("✅ 已完成：所有原本为 0 的标签已改为 1")