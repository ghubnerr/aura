#!/usr/bin/env python3
"""Generate a short VP8 `.ivf` stub for the WebRTC art track.

Requires ffmpeg with libvpx. Writes to AURA_IVF_DIR (default ./ivf_files).
Touching/updating this file triggers the Rust streamer's directory watcher.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("AURA_IVF_DIR", "./ivf_files"),
        help="Directory watched by the WebRTC streamer",
    )
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--name",
        default="aura_stub.ivf",
        help="Output filename inside out-dir",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH (need libvpx / VP8).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.name

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x480:rate=30",
        "-t",
        str(args.seconds),
        "-c:v",
        "libvpx",
        "-b:v",
        "800k",
        "-deadline",
        "good",
        "-cpu-used",
        "4",
        "-an",
        str(out_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Nudge mtime so notify watchers fire even if content was identical.
    now = time.time()
    os.utime(out_path, (now, now))
    print(f"Wrote {out_path.resolve()} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
