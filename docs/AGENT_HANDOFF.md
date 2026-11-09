# Aura agent handoff (cloud ↔ Windows)

**Branch / PR:** `cursor/cleanup-modernize-1aa4` → https://github.com/ghubnerr/aura/pull/43  
**Windows:** Gabes-PC @ `192.168.1.200`

## Fix for “NEUTRAL text / gray / emotion stuck”

You were still seeing the **old labeled stub IVF** (burned-in NEUTRAL + crop).  
Do a clean restart on the **morphing** continuous path (no text overlays):

```bat
:: kill old stack first
taskkill /FI "WINDOWTITLE eq aura-*" /F

git pull origin cursor/cleanup-modernize-1aa4
scripts\start_host.bat
```

Open `https://192.168.1.200:8443/?signaling=192.168.1.200:3031` and **Ctrl+Shift+R**.

Expect:
1. Colorful continuous motion on the left (full frame, letterboxed — not cropped text)
2. Emotion updates within ~1s of Capture Face (`[demo]` until a real checkpoint exists)
3. Art meta shows `MorphingStubBackend`

GPU diffusion (optional later): `scripts\start_gpu_host.bat` with free VRAM.
