import argparse
import os
from urllib.parse import urlparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 HTTP 视频流作为实时视频源进行作物识别。"
    )
    parser.add_argument(
        "--source",
        "-s",
        default="http://192.168.5.1",
        help="HTTP 视频流地址或摄像头 IP，例如 http://<camera-ip> 或 http://<camera-ip>/stream",
    )
    parser.add_argument(
        "--stream-path",
        default=None,
        help="可选的视频流路径，例如 /stream、/mjpeg/1；不填时会自动尝试常见路径",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=r"D:\作业\intelligent_agriculture\models\best.pt",
        help="YOLO 模型权重路径",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.80,
        help="检测置信度阈值，默认 0.80",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="不显示检测窗口，仅进行模型推理",
    )
    parser.set_defaults(show=True)
    return parser.parse_args()


def build_source_candidates(source, stream_path=None):
    if stream_path:
        stream_path = stream_path if stream_path.startswith("/") else f"/{stream_path}"

    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        base = f"{parsed.scheme}://{parsed.netloc}"
        candidates = [source]
        if parsed.path not in ("", "/"):
            candidates.append(base)
        else:
            candidates.append(base)

        if stream_path:
            candidates.append(base + stream_path)
        else:
            candidates.extend([
                # base + "/stream",
                # base + "/mjpeg/1",
                # base + "/jpg",
                base + "/capture",
            ])
        return candidates

    # 兼容直接传入 IP/主机名
    base = f"http://{source}"
    candidates = [base]
    if stream_path:
        candidates.append(base + stream_path)
    else:
        candidates.extend([
            base + "/stream",
            base + "/mjpeg/1",
            base + "/jpg",
            base + "/capture",
        ])
    return candidates


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    model = YOLO(args.model)
    candidates = build_source_candidates(args.source, args.stream_path)

    print(f"加载模型: {args.model}")
    print("尝试连接视频流...")

    last_error = None
    for url in candidates:
        print(f"尝试视频源: {url}")
        try:
            results = model.predict(
                source=url,
                conf=args.conf,
                show=args.show,
                stream=True,
            )
            print(f"成功使用视频源: {url}")
            print("实时识别已启动，按 'q' 键退出...")
            for _ in results:
                pass
            break
        except Exception as e:
            last_error = e
            print(f"{url} 连接失败: {e}")
    else:
        raise RuntimeError(f"无法连接到任何可用的视频流地址。最后错误：{last_error}")
