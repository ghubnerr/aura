# Aura Python Module

Built with Maturin (`maturin develop` inside `.venv` before testing anything that needs the Rust extension).

### Running WebRTC (LAN + camera)

Chrome blocks `ws://` from `https://` pages (mixed content) and blocks camera on plain LAN HTTP. Use this stack:

```bash
export AURA_BIND_IP=127.0.0.1
export AURA_SIGNALING_HOST=127.0.0.1
export AURA_SIGNALING_PORT=3030

# 1) Plain WS signaling (streamer + WSS proxy both connect here)
python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030

# 2) IVF → WebRTC streamer
python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030

# 3) WSS proxy for the browser (same self-signed cert as the UI)
python -m aura.webrtc.wss_proxy --port 3031

# 4) HTTPS UI
python scripts/serve_ui_https.py
```

Open (accept the cert warning once):

`https://<host-ip>:8443/?signaling=<host-ip>:3031`

Emotion → art MVP: face captures classify emotion and copy
`docs/samples/emotions/<emotion>.ivf` → `ivf_files/live.ivf`.

### Localhost-only demo (no TLS)

```bash
python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
python -m http.server 8080 --directory ui/static
# http://127.0.0.1:8080/
```

### Testing

```bash
pytest aura/tests
cargo test
```
