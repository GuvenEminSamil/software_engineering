import sys, csv, os, time
from pathlib import Path
from typing import cast
from transformers import pipeline
from transformers.pipelines import TextClassificationPipeline

HERE = Path(__file__).parent
DEFAULT_INP = HERE / "sample.csv"
DEFAULT_OUT = HERE / "predictions.csv"

inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INP
out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

print(f"[info] Input CSV:  {inp}")
print(f"[info] Output CSV: {out}")

if not inp.exists():
    print(f"[error] Input file not found: {inp}")
    print("[hint] Put your CSV next to main.py OR set Run Configuration → Parameters to point to your CSV.")
    sys.exit(1)

print("[info] Loading sentiment model (first run will download weights)...")
t0 = time.time()
clf: TextClassificationPipeline = cast(
    TextClassificationPipeline,
    pipeline(task="text-classification",
             model="distilbert-base-uncased-finetuned-sst-2-english")  # tiny & fast for first test
)
print(f"[info] Model ready in {time.time() - t0:.1f}s")

# Process CSV
with inp.open(newline="", encoding="utf-8") as f_in, out.open("w", newline="", encoding="utf-8") as f_out:
    r = csv.DictReader(f_in)
    if not r.fieldnames or "text" not in r.fieldnames:
        print("[error] CSV must contain a 'text' column.")
        sys.exit(2)
    w = csv.DictWriter(f_out, fieldnames=list(r.fieldnames) + ["label", "score"])
    w.writeheader()
    for i, row in enumerate(r, 1):
        pred = clf(row["text"])[0]
        row["label"], row["score"] = pred["label"], f'{pred["score"]:.4f}'
        w.writerow(row)
        if i % 5 == 0:
            print(f"[info] Processed {i} rows...")

print(f"[done] Wrote predictions → {out}")
