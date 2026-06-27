import argparse
import os
import time
from urllib.parse import urlparse

import cv2
import numpy as np
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


DISPLAY_HEIGHT = 500
PANEL_WIDTH = 290
PANEL_BG = (22, 22, 42)
PANEL_ACCENT = (0, 200, 200)
PANEL_TEXT = (230, 230, 230)
PANEL_DIM = (160, 160, 180)
LINE_HEIGHT = 32


def draw_detections(frame, result):
    """只在视频帧上绘制检测框和标签，不叠加任何信息条。"""
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
    return class_counts


def create_info_panel(panel_height):
    """创建深色背景的侧边信息面板。"""
    panel = np.full((panel_height, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
    # 左侧亮色分隔线
    cv2.line(panel, (0, 0), (0, panel_height - 1), PANEL_ACCENT, 2)
    return panel


def draw_info_panel(panel, source, fps, conf_threshold, class_counts, paused):
    """在侧边面板上绘制所有文字信息。"""
    h = panel.shape[0]
    x = 16
    y = 0

    # ── 标题 ──
    y += 30
    cv2.putText(panel, "Plant", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    y += 30
    cv2.putText(panel, "Recognition", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    # 彩色下划线
    y += 6
    cv2.line(panel, (x, y), (x + 130, y), PANEL_ACCENT, 3)

    # ── 分隔 ──
    y += 24
    cv2.line(panel, (x, y), (PANEL_WIDTH - 16, y), (60, 60, 80), 1)

    # ── 状态指示 ──
    y += 24
    status_color = (80, 200, 255) if not paused else (80, 80, 255)
    status_text = "LIVE" if not paused else "PAUSED"
    cv2.putText(panel, f"Status: {status_text}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # ── 基本信息 ──
    y += LINE_HEIGHT + 4
    cv2.putText(panel, "Source", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_DIM, 1)
    y += LINE_HEIGHT - 4
    # 长 URL 折行显示
    src_display = source if len(source) <= 35 else source[:32] + "..."
    cv2.putText(panel, src_display, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_TEXT, 1)

    y += LINE_HEIGHT + 4
    cv2.putText(panel, "Confidence", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_DIM, 1)
    y += LINE_HEIGHT - 4
    cv2.putText(panel, f">= {conf_threshold:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_TEXT, 1)

    y += LINE_HEIGHT + 4
    cv2.putText(panel, "FPS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_DIM, 1)
    y += LINE_HEIGHT - 4
    fps_color = (100, 255, 100) if fps >= 15 else (100, 200, 255) if fps >= 8 else (100, 100, 255)
    cv2.putText(panel, f"{fps:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2)

    # ── 分隔 ──
    y += 18
    cv2.line(panel, (x, y), (PANEL_WIDTH - 16, y), (60, 60, 80), 1)

    # ── 检测统计 ──
    y += 24
    cv2.putText(panel, "Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if class_counts:
        y += LINE_HEIGHT
        for label, count in sorted(class_counts.items()):
            color = get_class_color(label)
            # 小色块
            cv2.rectangle(panel, (x, y - 10), (x + 14, y + 2), color, -1)
            cv2.putText(panel, f"{label}: {count}", (x + 22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_TEXT, 1)
            y += LINE_HEIGHT
    else:
        y += LINE_HEIGHT
        cv2.putText(panel, "(none)", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_DIM, 1)

    # ── 操作提示（固定在面板底部） ──
    cv2.line(panel, (x, h - 56), (PANEL_WIDTH - 16, h - 56), (60, 60, 80), 1)
    cv2.putText(panel, "Controls", (x, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_DIM, 1)
    cv2.putText(panel, "Q - Quit", (x, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_TEXT, 1)
    cv2.putText(panel, "P - Pause/Play", (x + 100, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_TEXT, 1)


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
            class_counts = {}
            start_time = time.time()
            frame_count = 0
            window_name = "Plant Recognition"

            if args.show:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            first_frame = True

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
                    class_counts = draw_detections(frame, result)

                # 缩放视频帧到目标高度，保持宽高比
                h, w = frame.shape[:2]
                scale = DISPLAY_HEIGHT / h
                display_frame = cv2.resize(frame, (int(w * scale), DISPLAY_HEIGHT))

                # 创建侧边信息面板（高度与缩放后的视频一致）
                panel = create_info_panel(DISPLAY_HEIGHT)
                draw_info_panel(panel, url, fps, args.conf, class_counts, paused)

                # 拼接视频帧和面板
                display = np.hstack([display_frame, panel])

                if first_frame:
                    dh, dw = display.shape[:2]
                    cv2.resizeWindow(window_name, dw, dh)
                    first_frame = False

                cv2.imshow(window_name, display)
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
