"""Generate narration audio with piper TTS and a timing file for the manim scenes.

Run with the tts env:  /data/user_data/vawale/envs/tts/bin/python gen_audio.py --out <dir>
Writes <out>/<scene>_<idx>.wav and <out>/timing.json {scene: [duration_s, ...]}.
"""
import argparse, hashlib, json, os, re, wave
from narration import SCENES, PRON

VOICE = "/data/user_data/vawale/MimicKit/video/voices/en_US-ryan-high.onnx"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default=VOICE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from piper import PiperVoice
    voice = PiperVoice.load(a.voice)
    timing = {}
    for scene, lines in SCENES.items():
        timing[scene] = []
        for i, text in enumerate(lines):
            say = text
            for pat, rep in PRON:
                say = re.sub(pat, rep, say)
            path = os.path.join(a.out, f"{scene}_{i:02d}.wav")
            if not os.path.exists(path):
                with wave.open(path, "wb") as wf:
                    voice.synthesize_wav(say, wf)
            with wave.open(path, "rb") as wf:
                dur = wf.getnframes() / wf.getframerate()
            timing[scene].append(round(dur, 3))
            print(f"{scene}[{i}] {dur:6.2f}s  {text}")
    with open(os.path.join(a.out, "timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    total = sum(sum(v) for v in timing.values())
    print(f"total speech {total:.1f}s")

if __name__ == "__main__":
    main()
