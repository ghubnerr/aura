@echo off
REM Reliable Aura host: continuous morphing art + trained emotion CNN if present.
REM Use scripts\start_gpu_host.bat only when you explicitly want SD-Turbo.

cd /d %~dp0\..

if "%STORAGE_PATH%"=="" set STORAGE_PATH=.\output
set AURA_ART_BACKEND=morphing
set AURA_ART_FPS=20
set AURA_ART_CHUNK_FRAMES=20

echo Pulling latest...
git pull origin cursor/cleanup-modernize-1aa4

echo.
echo Verifying emotion checkpoint...
python scripts\verify_emotion_ckpt.py
if errorlevel 1 (
  echo WARNING: no trained checkpoint — UI will show Emotion ... [demo]
) else (
  echo Checkpoint OK — signaling should print: Emotion runtime: loaded checkpoint
)

echo.
echo Starting stack...
start "aura-signaling" cmd /k set STORAGE_PATH=%STORAGE_PATH%&& set AURA_ART_BACKEND=%AURA_ART_BACKEND%&& python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
timeout /t 3 >nul
start "aura-streamer" cmd /k set STORAGE_PATH=%STORAGE_PATH%&& python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
start "aura-wss" cmd /k python -m aura.webrtc.wss_proxy --port 3031
start "aura-https" cmd /k python scripts/serve_ui_https.py

echo.
echo Open: https://192.168.1.200:8443/?signaling=192.168.1.200:3031
echo Hard-refresh (Ctrl+Shift+R). Emotion should NOT say [demo] if the .pth loaded.
