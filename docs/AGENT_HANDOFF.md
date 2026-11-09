# Aura agent handoff (cloud ↔ Windows)

**Cloud agent:** https://cursor.com/agents/bc-743c991d-d8ea-4980-897f-9c1ebb231aa4  
**Branch / PR:** `cursor/cleanup-modernize-1aa4` → https://github.com/ghubnerr/aura/pull/43  
**Windows worker:** `Gabes-PC` (`c:\Users\gabri\Projects\aura`) @ `192.168.1.200`

Cloud inbound TCP is blocked, so **WebRTC host + GPU work run on Windows**. Cloud owns repo cleanup, architecture, and PR iteration.

## Live host commands (current)

```bat
git pull origin cursor/cleanup-modernize-1aa4
maturin develop

python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
python -m aura.webrtc.wss_proxy --port 3031
python scripts/serve_ui_https.py
```

Open: `https://192.168.1.200:8443/?signaling=192.168.1.200:3031`

Emotion captures publish `docs/samples/emotions/<emotion>.ivf` → `ivf_files/live.ivf`.

## Shared mailbox

Use **PR #43 comments** for status so both agents can read without a direct DM API.

## Success criteria (MVP)

1. Browser on LAN sees camera + remote art track (emotion stub IVF)
2. Face capture → emotion embedding → live IVF swap
3. Next: replace stub publisher with StreamDiffusionV2/Krea realtime on the 5060 Ti
