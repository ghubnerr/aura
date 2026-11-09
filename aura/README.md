# Aura Python Module

Built with Maturin (`maturin develop` inside `.venv` before testing anything that needs the Rust extension).

### Running WebRTC

On the GPU/cloud host (bind all interfaces so LAN/remote clients can connect):

```bash
export AURA_BIND_IP=0.0.0.0
export AURA_SIGNALING_HOST=127.0.0.1   # streamer joins locally
export AURA_SIGNALING_PORT=3030

python -m aura.webrtc.signaling --bind-ip 0.0.0.0 --port 3030
python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
```

Serve the UI (from another terminal):

```bash
python -m http.server 8080 --directory ui/static
# Preferred on LAN (camera needs a secure context):
python scripts/serve_ui_https.py
```

Open:
- `http://<host-ip>:8080/?signaling=<host-ip>:3030` or
- `https://<host-ip>:8443/?signaling=<host-ip>:3030` (accept self-signed cert)

Seed a stub art track so the remote video shows pixels:

```bash
mkdir -p ivf_files
cp docs/samples/aura_stub.ivf ivf_files/
# or: python scripts/make_stub_ivf.py
```

### Testing

```bash
pytest aura/tests
cargo test
```
