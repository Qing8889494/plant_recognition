from ultralytics import YOLO

# 加载训练好的模型
model = YOLO(r"D:\作业\intelligent_agriculture\models\best_V4.pt")

# 调用摄像头实时识别（source=0 为默认摄像头）
# show=True   → 实时弹出窗口显示检测结果
# stream=True → 流式处理，逐帧返回（配合 for 循环持续运行）
results = model.predict(
    source=0,
    conf=0.80,
    show=True,
    stream=True
)

print("实时识别已启动，按 'q' 键退出...")

# 遍历结果流，保持程序运行
for r in results:
    pass  # 帧已在 show=True 时自动显示
