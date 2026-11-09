# Continuous emotion-conditioned art

## Product contract

```
next_frames = backend.step(current_frame, emotion_embedding)
```

- **Input:** current art frame `x_t` + emotion embedding `e_t`
- **Output:** next frames continuing from `x_t`, conditioned on `e_t`
- Mood updates change `e_t` only — temporal state is not reset

## Backends (Gabes-PC / RTX 5060 Ti 16GB)

| `AURA_ART_BACKEND` | Notes |
|---|---|
| `diffusers_turbo` | **Recommended.** SD-Turbo img2img; ~4–10 FPS @ 512² in fp16 |
| `streamdiffusion` | Optional if `pip install streamdiffusion` |
| `morphing` | CPU stub (no GPU) |
| `auto` | CUDA → diffusers_turbo, else morphing |

StreamDiffusionV2 / Krea Realtime need ~24–32GB VRAM — too large for 16GB today.
Same `step()` API when VRAM allows.

### Emotion → prompt

The 2048-d embedding selects the emotion label; that label maps to an abstract
art prompt. Denoise **strength** (~0.42) keeps the previous frame as init image
so the stream stays continuous. Embedding energy nudges strength slightly.

## Host

```bat
scripts\start_gpu_host.bat
```

Or manually:

```bat
set AURA_ART_BACKEND=diffusers_turbo
python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
python -m aura.webrtc.wss_proxy --port 3031
python scripts/serve_ui_https.py
```

First signaling start downloads `stabilityai/sd-turbo` and warms the pipeline.
