# Aura agent handoff (cloud ↔ Windows)

**Cloud agent:** https://cursor.com/agents/bc-743c991d-d8ea-4980-897f-9c1ebb231aa4  
**Branch / PR:** `cursor/cleanup-modernize-1aa4` → https://github.com/ghubnerr/aura/pull/43  
**Windows worker:** `Gabes-PC` (`c:\Users\gabri\Projects\aura`) @ `192.168.1.200`

## Critical fix (pull this)

Streamer previously **waited for a file Modify** before sending any frames, and **stopped at IVF EOF**. That left the browser with an empty remote track — users only saw their webcam / face capture.

Now the streamer:
1. Seeds from existing `ivf_files/live.ivf` immediately
2. Loops frames from memory forever
3. Reloads when emotion publish replaces `live.ivf`
4. Trickles ICE candidates back to the browser

**Must rebuild Rust:** `maturin develop`

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

Hard-refresh the browser (Ctrl+Shift+R). After **Start Call**, the large left panel is the art stream; the small right panel is your camera.

## Success criteria

1. Start Call → continuous abstract art on the left (not your face)
2. Capture Face → emotion label updates; art palette/motion changes
3. Next: replace stub IVF publisher with StreamDiffusionV2/Krea on the 5060 Ti
