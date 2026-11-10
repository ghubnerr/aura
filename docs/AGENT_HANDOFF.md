# Aura agent handoff (cloud ↔ Windows)

**Branch / PR:** `cursor/cleanup-modernize-1aa4` → https://github.com/ghubnerr/aura/pull/43  
**Windows:** Gabes-PC @ `192.168.1.200`

## Emotion CNN trained — load it live

Assuming `output\aura_storage\aura_emotion_classifier.pth` exists on Gabes-PC:

```bat
taskkill /FI "WINDOWTITLE eq aura-*" /F
cd /d c:\Users\gabri\Projects\aura
git pull origin cursor/cleanup-modernize-1aa4
set STORAGE_PATH=.\output
scripts\start_host.bat
```

Verify:
1. `python scripts\verify_emotion_ckpt.py` → `OK: checkpoint loads`
2. Signaling log: `Emotion runtime: loaded checkpoint ...` (no demo heuristic)
3. UI Emotion updates **without** `[demo]` after Capture Face

Please paste best val acc / `emotion_train_metrics.json` on PR #43 if not already.

## Next

1. CLIP emotion↔art alignment (`python -m aura.clip.train`)
2. Condition SD-Turbo on the embedding (LoRA / IP-Adapter)
