/**
 * Browser WebRTC client for Aura.
 *
 * On HTTPS pages the client uses wss:// automatically (point at the WSS proxy).
 * Override host with ?signaling=host:port — e.g. 192.168.1.200:3031
 */
(function () {
  const params = new URLSearchParams(window.location.search);
  const signalingParam = params.get("signaling");
  const defaultHost = window.location.hostname || "127.0.0.1";
  const defaultPort = window.location.protocol === "https:" ? "3031" : "3030";
  const signalingTarget = signalingParam || `${defaultHost}:${defaultPort}`;
  const forcePlain = params.get("tls") === "0";
  const useTls =
    !forcePlain &&
    (window.location.protocol === "https:" || params.get("tls") === "1");
  const scheme = useTls ? "wss" : "ws";
  const signalingUrl = `${scheme}://${signalingTarget}/signaling`;

  const localVideo = document.getElementById("localVideo");
  const remoteVideo = document.getElementById("remoteVideo");
  const capturedImage = document.getElementById("capturedImage");
  const statusEl = document.getElementById("status");
  const emotionEl = document.getElementById("emotion");
  const artMetaEl = document.getElementById("artMeta");

  remoteVideo.muted = true;
  remoteVideo.setAttribute("playsinline", "true");
  remoteVideo.setAttribute("autoplay", "true");

  const signalingSocket = new WebSocket(signalingUrl);
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
    console.log(text);
  }

  function setEmotion(text) {
    if (!emotionEl) return;
    emotionEl.innerHTML = `Emotion: <strong>${text}</strong>`;
  }

  function setArtMeta(text) {
    if (!artMetaEl) return;
    artMetaEl.innerHTML = `Art: <strong>${text}</strong>`;
  }

  function sendMessage(data) {
    if (signalingSocket.readyState !== WebSocket.OPEN) {
      console.error("WebSocket is not open. Cannot send message:", data);
      return;
    }
    signalingSocket.send(JSON.stringify(data));
  }

  async function playRemote() {
    try {
      await remoteVideo.play();
    } catch (err) {
      console.warn("remoteVideo.play() blocked:", err);
      setStatus("Tap the page once if art is paused (autoplay)");
    }
  }

  function syncRemoteAspect() {
    if (!remoteVideo.videoWidth || !remoteVideo.videoHeight) return;
    remoteVideo.style.aspectRatio = `${remoteVideo.videoWidth} / ${remoteVideo.videoHeight}`;
    setArtMeta(`${remoteVideo.videoWidth}x${remoteVideo.videoHeight}`);
  }

  remoteVideo.addEventListener("loadedmetadata", syncRemoteAspect);
  remoteVideo.addEventListener("resize", syncRemoteAspect);

  function captureFrame() {
    if (!localVideo.videoWidth || !localVideo.videoHeight) {
      console.warn("Local video not ready for capture");
      setEmotion("camera not ready");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = localVideo.videoWidth;
    canvas.height = localVideo.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      console.error("Failed to get canvas context");
      return;
    }
    context.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL("image/png");
    capturedImage.src = imageData;
    capturedImage.style.display = "block";
    setEmotion("sending…");
    sendMessage({ type: "image", data: imageData });
  }

  async function startLocalMedia() {
    const localStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    localVideo.srcObject = localStream;
    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));
  }

  setStatus(`Connecting to ${signalingUrl}…`);

  signalingSocket.onopen = () => {
    setStatus(`Connected to ${signalingUrl}`);
    document.getElementById("startCall").disabled = false;
    document.getElementById("captureImage").disabled = false;
  };

  signalingSocket.onerror = () => {
    setStatus(
      useTls
        ? `WSS error — is the proxy up on ${signalingTarget}?`
        : `WebSocket error — is signaling up on ${signalingTarget}?`
    );
  };

  signalingSocket.onclose = (event) => {
    setStatus(`WebSocket closed: ${event.code} ${event.reason || ""}`.trim());
  };

  document.getElementById("startCall").onclick = async () => {
    if (signalingSocket.readyState !== WebSocket.OPEN) {
      setStatus("WebSocket is not open, cannot start call");
      return;
    }
    try {
      await startLocalMedia();
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      sendMessage({ type: "offer", sdp: offer.sdp });
      setStatus("Sent offer — waiting for art track");
    } catch (error) {
      console.error(error);
      setStatus(`Camera/WebRTC error: ${error.message || error}`);
    }
  };

  pc.onicecandidate = ({ candidate }) => {
    if (!candidate) return;
    sendMessage({
      type: "candidate",
      candidate: candidate.candidate,
      sdpMid: candidate.sdpMid,
      sdpMLineIndex: candidate.sdpMLineIndex,
    });
  };

  pc.onconnectionstatechange = () => {
    setStatus(`WebRTC: ${pc.connectionState}`);
  };

  signalingSocket.onmessage = async (message) => {
    let data;
    try {
      data = JSON.parse(message.data);
    } catch (err) {
      console.warn("Non-JSON signaling message", err);
      return;
    }

    if (data.type === "candidate") {
      if (data.candidate == null || data.candidate === "") return;
      try {
        await pc.addIceCandidate(
          new RTCIceCandidate({
            candidate: data.candidate,
            sdpMid: data.sdpMid ?? data.sdp_mid ?? null,
            sdpMLineIndex: data.sdpMLineIndex ?? data.sdp_mline_index ?? null,
          })
        );
      } catch (error) {
        console.error("Error adding received ICE candidate", error);
      }
      return;
    }

    if (data.type === "offer") {
      try {
        await pc.setRemoteDescription(new RTCSessionDescription(data));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        sendMessage({ type: "answer", sdp: answer.sdp });
        setStatus("Answered remote offer");
      } catch (error) {
        console.error("Error handling received offer", error);
      }
      return;
    }

    if (data.type === "answer") {
      try {
        await pc.setRemoteDescription(new RTCSessionDescription(data));
        setStatus("Remote answer applied — waiting for art frames");
      } catch (error) {
        console.error("Error setting remote description from answer", error);
      }
      return;
    }

    if (data.type === "triggerimagecapture") {
      captureFrame();
      return;
    }

    if (data.type === "emotion") {
      if (data.error) {
        setEmotion(`${data.emotion} (${data.error})`);
      } else if (data.emotion) {
        const suffix = data.demo ? " [demo]" : "";
        setEmotion(`${data.emotion}${suffix}`);
      }
      return;
    }

    if (data.type === "art_status") {
      const bits = [data.backend, data.fps ? `${data.fps}fps` : null].filter(Boolean);
      setArtMeta(bits.join(" · "));
    }
  };

  pc.ontrack = (event) => {
    const stream = event.streams[0] || new MediaStream([event.track]);
    remoteVideo.srcObject = stream;
    setStatus("Receiving remote art stream");
    playRemote();
    syncRemoteAspect();
    event.track.onunmute = () => {
      playRemote();
      syncRemoteAspect();
    };
  };

  document.getElementById("captureImage").addEventListener("click", captureFrame);
})();
