#!/usr/bin/env python3
"""Generate distinct abstract VP8 stubs per emotion (not ffmpeg test bars).

Burns the emotion name into each clip so mood swaps are obvious in the UI.
Requires ffmpeg with libvpx.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Visual recipe per emotion: lavfi source + hue + label color.
# Animated hue expressions keep the clip clearly "video", not a still.
EMOTIONS = {
    "happy": {
        "src": "gradients=size=640x480:rate=30:c0=0xffb703:c1=0xfb8500:x0=0:y0=0:x1=640:y1=480:speed=0.03",
        "vf": "hue=h='20*t':s=1.4,eq=saturation=1.3:brightness=0.05",
    },
    "sad": {
        "src": "gradients=size=640x480:rate=30:c0=0x023e8a:c1=0x48cae4:x0=0:y0=480:x1=640:y1=0:speed=0.015",
        "vf": "hue=h='200+8*t':s=0.7,eq=saturation=0.8:brightness=-0.05",
    },
    "anger": {
        "src": "gradients=size=640x480:rate=30:c0=0x6a040f:c1=0xdc2f02:x0=0:y0=0:x1=320:y1=480:speed=0.06",
        "vf": "hue=h='-15+25*sin(2*PI*t)':s=1.6,eq=contrast=1.2:saturation=1.4",
    },
    "fear": {
        "src": "gradients=size=640x480:rate=30:c0=0x240046:c1=0x7b2cbf:x0=640:y0=0:x1=0:y1=480:speed=0.04",
        "vf": "hue=h='270+15*t':s=1.1,eq=brightness=-0.1:saturation=1.1",
    },
    "surprise": {
        "src": "gradients=size=640x480:rate=30:c0=0xfff000:c1=0xff006e:x0=0:y0=0:x1=640:y1=240:speed=0.08",
        "vf": "hue=h='40*t':s=1.5,eq=saturation=1.5:contrast=1.1",
    },
    "disgust": {
        "src": "gradients=size=640x480:rate=30:c0=0x1b4332:c1=0x95d5b2:x0=0:y0=0:x1=480:y1=480:speed=0.02",
        "vf": "hue=h='100+10*t':s=0.9,eq=saturation=0.9",
    },
    "contempt": {
        "src": "gradients=size=640x480:rate=30:c0=0x3d405b:c1=0x81b29a:x0=320:y0=0:x1=320:y1=480:speed=0.02",
        "vf": "hue=h='140+5*t':s=0.6,eq=saturation=0.7:brightness=-0.02",
    },
    "neutral": {
        "src": "gradients=size=640x480:rate=30:c0=0x212529:c1=0xadb5bd:x0=0:y0=0:x1=640:y1=480:speed=0.01",
        "vf": "hue=h='5*t':s=0.35,eq=saturation=0.4",
    },
}

# Fallback when `gradients` is unavailable (older ffmpeg).
FALLBACK_SRC = "testsrc2=size=640x480:rate=30"


def _have_gradients() -> bool:
    probe = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "gradients=size=16x16:rate=1:c0=red:c1=blue",
            "-t",
            "0.1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def render(emotion: str, out_path: Path, seconds: float, use_gradients: bool) -> None:
    recipe = EMOTIONS[emotion]
    src = recipe["src"] if use_gradients else FALLBACK_SRC
    # Always burn the emotion name so swaps are obvious.
    vf = (
        f"format=yuv420p,{recipe['vf']},"
        f"drawtext=text='{emotion.upper()}':fontsize=42:fontcolor=white:"
        f"borderw=3:bordercolor=black:x=24:y=24,"
        f"drawtext=text='aura stub':fontsize=18:fontcolor=white@0.7:"
        f"x=24:y=72"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        src,
        "-t",
        str(seconds),
        "-vf",
        vf,
        "-c:v",
        "libvpx",
        "-b:v",
        "1M",
        "-deadline",
        "good",
        "-cpu-used",
        "4",
        "-an",
        str(out_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parents[1] / "docs" / "samples" / "emotions"
        ),
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH (need libvpx / VP8).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_gradients = _have_gradients()
    if not use_gradients:
        print("ffmpeg gradients filter missing; falling back to testsrc2+hue", file=sys.stderr)

    for emotion in EMOTIONS:
        out_path = out_dir / f"{emotion}.ivf"
        render(emotion, out_path, args.seconds, use_gradients)
        print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
