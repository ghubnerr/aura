# Aura agent handoff (cloud ↔ Windows)

**Cloud agent:** https://cursor.com/agents/bc-743c991d-d8ea-4980-897f-9c1ebb231aa4  
**Branch / PR:** `cursor/cleanup-modernize-1aa4` → https://github.com/ghubnerr/aura/pull/43  
**Windows worker:** `Gabes-PC` (`c:\Users\gabri\Projects\aura`) @ `192.168.1.200`

## Continuous generation contract

```
next_frames = step(current_frame, emotion_embedding)
```

Signaling starts `ContinuousArtEngine` (CPU morphing stub for now). Face
captures update the conditioning embedding; the frame state is **not** reset.
See `docs/CONTINUOUS_ART.md`. Clip IVF swapping is fallback-only.

## Live host commands

```bat
git pull origin cursor/cleanup-modernize-1aa4
maturin develop

python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
python -m aura.webrtc.wss_proxy --port 3031
python scripts/serve_ui_https.py
```

Open: `https://192.168.1.200:8443/?signaling=192.168.1.200:3031`  
Hard-refresh after pulls.

## Success criteria

1. Start Call → continuous art on the left (mood drifts, no hard-cut playlist)
2. Capture Face → emotion status updates; stream keeps temporal continuity
3. Swap `MorphingStubBackend` for StreamDiffusionV2/Krea with the same `step()` API
