import os
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("D:\作业\intelligent_agriculture\models\\best_V4.pt")   # 如果不在同一目录，写上完整路径，如 "C:/Users/.../best.pt"

# 图片文件夹路径（替换为你的图片文件夹路径）
image_dir = r"D:\\作业\\intelligent_agriculture\\test1"
output_dir = os.path.join(image_dir, "result")

# 对文件夹中的所有图片进行检测，结果保存到 result 子目录
results = model(
    source=image_dir,
    conf=0.80,
    # show=True,
    save=True,
    project=output_dir,
    name=""          # name 为空避免再嵌套一层子目录
)

print(f"结果图片已保存到：{output_dir}")

# 如果需要逐张处理，可以遍历文件夹中的图片
# import glob
# for img_path in glob.glob(os.path.join(image_dir, "*.[jJpP][pPnN][gG]")) + \
#                    glob.glob(os.path.join(image_dir, "*.[jJ][pP][eE][gG]")) + \
#                    glob.glob(os.path.join(image_dir, "*.[pP][nN][gG]")):
#     results = model(img_path, conf=0.8, save=True, project=output_dir, name="")
