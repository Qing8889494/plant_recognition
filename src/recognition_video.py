import os
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("best.pt")   # 如果不在同一目录，写上完整路径，如 "C:/Users/.../best.pt"

# 视频文件路径（替换为你的视频文件路径）
video_path = r"C:\\Users\\ASUS\\Desktop\\testVideo.mp4"
output_dir = os.path.dirname(video_path)
output_name = os.path.splitext(os.path.basename(video_path))[0] + "_result"

# 对视频文件进行检测，实时显示结果并保存输出视频到同目录
results = model(
    source=video_path,
    conf=0.8,
    # show=True,
    save=True,
    project=output_dir,
    name=output_name
)

print(f"结果视频已保存到：{os.path.join(output_dir, output_name)}")

# 如果需要逐帧处理，可以使用 results 返回的帧结果
# for res in results:
#     res.show()
#     # res.save("frame_result.jpg")  # 保存每帧结果