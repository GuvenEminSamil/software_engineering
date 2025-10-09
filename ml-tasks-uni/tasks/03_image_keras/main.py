from pathlib import Path
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array

HERE = Path(__file__).parent
DEFAULT_IMG = HERE / "images" / "sample_image.jpg"

def classify(img_path: Path):
    print(f"[info] Using image: {img_path}")
    if not img_path.exists():
        print("[error] Image not found. Put a file in images/ or set Parameters.")
        raise SystemExit(1)

    print("[info] Loading MobileNetV2 (ImageNet) …")
    model = MobileNetV2(weights="imagenet")

    img = load_img(str(img_path), target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    top5 = decode_predictions(preds, top=5)[0]
    print("[result] Top-5 predictions:")
    for rank, (_, label, score) in enumerate(top5, 1):
        print(f"  {rank}. {label:20s}  {score:.4f}")

if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG
    classify(p)
