@echo off
REM Aura GPU host stack on Gabes-PC (RTX 5060 Ti).
REM Continuous art: next_frames = step(current_frame, emotion_embedding)
REM Backend: SD-Turbo img2img (fits 16GB). StreamDiffusionV2 needs ~24GB.

cd /d %~dp0\..

set AURA_ART_BACKEND=diffusers_turbo
set AURA_DIFFUSION_MODEL=stabilityai/sd-turbo
set AURA_DIFFUSION_STEPS=2
set AURA_DIFFUSION_STRENGTH=0.42
set AURA_ART_FPS=6
set AURA_ART_CHUNK_FRAMES=6

echo Pull + rebuild Rust bindings if needed...
git pull origin cursor/cleanup-modernize-1aa4
maturin develop

echo Starting signaling (loads SD-Turbo on first warmup — be patient)...
start "aura-signaling" cmd /k python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
timeout /t 8 >nul
start "aura-streamer" cmd /k python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
start "aura-wss" cmd /k python -m aura.webrtc.wss_proxy --port 3031
start "aura-https" cmd /k python scripts/serve_ui_https.py

echo Open: https://192.168.1.200:8443/?signaling=192.168.1.200:3031
echo Hard-refresh the browser after the signaling window prints warmup ok.
