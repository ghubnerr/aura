@echo off
REM Reliable Aura host: continuous morphing art (no GPU required).
REM Use scripts\start_gpu_host.bat only when you explicitly want SD-Turbo.

cd /d %~dp0\..

set AURA_ART_BACKEND=morphing
set AURA_ART_FPS=20
set AURA_ART_CHUNK_FRAMES=20

echo Pulling latest...
git pull origin cursor/cleanup-modernize-1aa4

echo Starting stack (morphing continuous art, no burned-in text)...
start "aura-signaling" cmd /k python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
timeout /t 3 >nul
start "aura-streamer" cmd /k python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
start "aura-wss" cmd /k python -m aura.webrtc.wss_proxy --port 3031
start "aura-https" cmd /k python scripts/serve_ui_https.py

echo.
echo Open: https://192.168.1.200:8443/?signaling=192.168.1.200:3031
echo Hard-refresh (Ctrl+Shift+R). Left panel should be colorful motion, no NEUTRAL text.
echo Emotion should flip within ~1s of Capture Face (demo heuristic until checkpoint exists).
