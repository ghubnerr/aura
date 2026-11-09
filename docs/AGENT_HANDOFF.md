# Aura agent handoff (cloud ↔ Windows)

**Branch / PR:** `cursor/cleanup-modernize-1aa4` → https://github.com/ghubnerr/aura/pull/43  
**Windows worker:** `Gabes-PC` @ `192.168.1.200`

## Fire: GPU continuous art

Contract: `next_frames = step(current_frame, emotion_embedding)`

On 16GB (5060 Ti) use **SD-Turbo img2img**, not StreamDiffusionV2 (~24GB).

```bat
git pull origin cursor/cleanup-modernize-1aa4
maturin develop
scripts\start_gpu_host.bat
```

Or:

```bat
set AURA_ART_BACKEND=diffusers_turbo
python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
python -m aura.webrtc.wss_proxy --port 3031
python scripts/serve_ui_https.py
```

Open: `https://192.168.1.200:8443/?signaling=192.168.1.200:3031`  
Wait for signaling log: `ContinuousArtEngine warmup ok (DiffusersTurboBackend)`.

## Verify

1. Left panel shows continuous abstract diffusion video (not test bars / face)
2. Capture Face → emotion label updates; art drifts without hard cut
3. Post result on PR #43

See `docs/CONTINUOUS_ART.md`.
