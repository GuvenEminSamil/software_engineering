from pathlib import Path
import argparse
from faster_whisper import WhisperModel

HERE = Path(__file__).parent
DEFAULT_AUDIO = HERE / "samples" / "Test.wav"

parser = argparse.ArgumentParser()
parser.add_argument("audio", nargs="?", help="Path to an audio file")
args = parser.parse_args()

audio_path = Path(args.audio) if args.audio else DEFAULT_AUDIO
print(f"[info] Using audio: {audio_path}")
if not audio_path.exists():
    print("[error] Audio file not found.")
    raise SystemExit(1)

model = WhisperModel("base", compute_type="int8")
segments, info = model.transcribe(str(audio_path))
text = " ".join(seg.text for seg in segments)
print({"language": info.language, "text": text})
