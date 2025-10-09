from pathlib import Path
import cv2
from ultralytics import YOLO

HERE = Path(__file__).parent
DEFAULT_IN = HERE / "input.mp4"
DEFAULT_OUT = HERE / "output.mp4"

def process_video(inp: Path, out: Path):
    print(f"[info] Input:  {inp}")
    print(f"[info] Output: {out}")
    if not inp.exists():
        print("[error] Input video not found. Put a file next to main.py or set Parameters.")
        raise SystemExit(1)

    print("[info] Loading YOLOv8n (first run downloads weights)…")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        print("[error] Could not open input video.")
        raise SystemExit(2)

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        print("[error] Could not open VideoWriter. Try changing FOURCC to 'avc1' or 'XVID'.")
        raise SystemExit(3)

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        writer.write(annotated)
        if frame_i % 10 == 0:
            print(f"[info] Processed {frame_i} frames…")

    cap.release()
    writer.release()
    print(f"[done] Wrote → {out}")

if __name__ == "__main__":
    import sys
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    process_video(inp, out)
