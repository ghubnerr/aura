@echo off
REM Train AffectNet emotion CNN on Gabes-PC (RTX 5060 Ti).
REM Needs: CUDA torch, kagglehub auth (KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json)

cd /d %~dp0\..

git pull origin cursor/cleanup-modernize-1aa4

echo.
echo === Emotion CNN training ===
echo Checkpoint will be written to %%STORAGE_PATH%%\aura_storage\aura_emotion_classifier.pth
echo (default STORAGE_PATH=.\output)
echo.

if "%STORAGE_PATH%"=="" set STORAGE_PATH=.\output

python -m aura.cnn.train --epochs 5 --batch-size 32 --split 0.85 --max-per-class 2000
if errorlevel 1 (
  echo Training failed. Check Kaggle credentials and CUDA torch.
  exit /b 1
)

echo.
echo Training finished. Restart signaling to load the checkpoint:
echo   set AURA_ART_BACKEND=morphing
echo   python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
echo Look for: Emotion runtime: loaded checkpoint ... (no demo_heuristic)
pause
