# Continuous emotion-conditioned art

## Product contract

The art stream is a **continuous video**, not a playlist of clips.

```
face crop ──► EmotionModel.embed ──► embedding e_t
                                          │
current art frame x_t ────────────────────┼──► backend.step(x_t, e_t) ──► x_{t+1..t+k}
                                          │
                                     WebRTC track (looping IVF chunks for now)
```

- **Input:** current image/frame `x_t` + conditioned emotion embedding `e_t`
- **Output:** next frames `x_{t+1}`, … conditioned on that pair
- When the viewer’s mood changes, `e_t` updates; `x_t` is **not** reset. Mood
  drifts through the existing temporal state.

## Backends

| Backend | Role |
|---|---|
| `MorphingStubBackend` | CPU stand-in that morphs color/motion continuously |
| StreamDiffusionV2 / Krea Realtime / Self-Forcing Wan | GPU realtime diffusion with the same `step(frame, embedding)` API |

Clip-based `EmotionArtGenerator.publish(emotion)` is **demo fallback only**.

## Host loop

`ContinuousArtEngine` runs on the GPU host (Gabes-PC):

1. Keep `current_frame` in memory
2. On each capture, `set_conditioning(embedding, emotion)`
3. Every tick: `frames = backend.step(current_frame, embedding)`
4. Encode a short VP8 chunk → `ivf_files/live.ivf` for the Rust streamer

Later we can push samples directly into the WebRTC track and drop IVF files.
