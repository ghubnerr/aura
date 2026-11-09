# Aura agent handoff (cloud ↔ Windows)

**Cloud agent:** https://cursor.com/agents/bc-743c991d-d8ea-4980-897f-9c1ebb231aa4  
**Branch / PR:** `cursor/cleanup-modernize-1aa4` → https://github.com/ghubnerr/aura/pull/43  
**Windows worker seen online:** `Gabes-PC` (`c:\Users\gabri\Projects\aura`)

Cloud inbound TCP is blocked, so **WebRTC host + GPU work run on Windows**. Cloud owns repo cleanup, architecture, and PR iteration.

## Split of work

### Windows agent (Gabes-PC / local Cursor)

1. `git fetch origin && git checkout cursor/cleanup-modernize-1aa4 && git pull`
2. Create `.venv`, `maturin develop`, install deps (see `aura/README.md`)
3. Confirm GPU: `nvidia-smi` (RTX 5060)
4. Collect LAN IP: `ipconfig` → note IPv4
5. Open firewall for **3030** and **8080**
6. Run host:
   - `python -m aura.webrtc.signaling --bind-ip 0.0.0.0 --port 3030`
   - `python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030`
   - `python -m http.server 8080 --directory ui/static`
7. Browser on same LAN: `http://<LAN_IP>:8080/?signaling=<LAN_IP>:3030`
8. Report back in PR #43 or chat: LAN IP, `maturin` OK/fail, WebRTC offer/answer status, GPU VRAM free

Later (after host works): emotion checkpoint load, Wan pair-gen smoke on 5060, StreamDiffusionV2/Krea realtime path.

### Cloud agent (this PR)

- Keep Python/Rust readable; fix CI
- Wire emotion → conditioning → IVF write path
- Dataset/CLIP/Wan scaffolding; docs
- Review Windows agent findings and merge fixes into the same branch

## Shared mailbox

Use **PR #43 comments** for status so both agents can read without a direct DM API.

## Success criteria (MVP)

1. Browser on LAN sees local camera + remote art track (even a looping IVF stub)
2. Face capture → signaling → `EmotionArtRuntime` prints emotion
3. Plan locked for realtime generator (StreamDiffusionV2/Krea vs StyleGAN-V)
