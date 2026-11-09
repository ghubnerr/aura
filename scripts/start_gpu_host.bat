@echo off
REM Optional GPU path. Prefer scripts\start_host.bat unless VRAM is free.
REM SD-Turbo img2img: step(current_frame, emotion_embedding). Needs ~6-10GB.

cd /d %~dp0\..

set AURA_ART_BACKEND=diffusers_turbo
set AURA_DIFFUSION_MODEL=stabilityai/sd-turbo
set AURA_DIFFUSION_STEPS=2
set AURA_DIFFUSION_STRENGTH=0.42
set AURA_ART_FPS=6
set AURA_ART_CHUNK_FRAMES=6

echo Pull...
git pull origin cursor/cleanup-modernize-1aa4

echo Starting GPU stack (first boot downloads sd-turbo)...
start "aura-signaling" cmd /k python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
timeout /t 5 >nul
start "aura-streamer" cmd /k python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
start "aura-wss" cmd /k python -m aura.webrtc.wss_proxy --port 3031
start "aura-https" cmd /k python scripts/serve_ui_https.py

echo Open: https://192.168.1.200:8443/?signaling=192.168.1.200:3031
echo If art stays gray/texty or VRAM OOMs, kill this stack and use start_host.bat.
