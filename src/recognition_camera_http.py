import argparse
import os
import time
from urllib.parse import urlparse

import cv2
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
                base + "/capture",
            ])
        return candidates

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


def get_class_color(label):
    palette = {
        "strawberry": (0, 165, 255),
        "pepper": (0, 255, 0),
        "tomato": (255, 0, 0),
    }
    return palette.get(label.lower(), (255, 255, 255))


def draw_overlay(frame, result, source, fps, conf_threshold):
    height, width = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (10, 10), (width - 10, 170), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        "Plant Recognition",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Source: {source}",
        (24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        f"Conf >= {conf_threshold:.2f}",
        (24, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (24, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )

    class_counts = {}
    if result.boxes:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names.get(cls_id, str(cls_id))
            class_counts[label] = class_counts.get(label, 0) + 1
            color = get_class_color(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    y_start = height - 70
    cv2.rectangle(frame, (10, y_start), (width - 10, height - 10), (20, 20, 20), -1)
    cv2.putText(frame, "Detected:", (24, y_start + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    x = 120
    for label, count in class_counts.items():
        color = get_class_color(label)
        cv2.putText(frame, f"{label}: {count}", (x, y_start + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        x += 110

    cv2.putText(frame, "q: quit  p: pause", (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def run_inference(args):
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
                stream=True,
            )
            print(f"成功使用视频源: {url}")
            print("实时识别已启动，按 'q' 键退出，按 'p' 暂停/继续...")

            paused = False
            start_time = time.time()
            frame_count = 0
            window_name = "Plant Recognition"

            if args.show:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 720)

            for result in results:
                if not args.show:
                    continue

                frame = result.orig_img
                if frame is None:
                    continue

                frame_count += 1
                if frame_count == 1:
                    start_time = time.time()
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0

                if not paused:
                    draw_overlay(frame, result, url, fps, args.conf)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('p'):
                    paused = not paused

            if args.show:
                cv2.destroyAllWindows()
            break
        except Exception as e:
            last_error = e
            print(f"{url} 连接失败: {e}")
    else:
        raise RuntimeError(f"无法连接到任何可用的视频流地址。最后错误：{last_error}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
