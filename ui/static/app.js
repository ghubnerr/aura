/**
 * Browser WebRTC client for Aura.
 *
 * Signaling host defaults to the page hostname. Override with ?signaling=host:port.
 * On HTTPS pages the client uses wss:// automatically (point at the WSS proxy).
 */
(function () {
  const params = new URLSearchParams(window.location.search);
  const signalingParam = params.get("signaling");
  const defaultHost = window.location.hostname || "127.0.0.1";
  const signalingTarget = signalingParam || `${defaultHost}:3030`;
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

  const signalingSocket = new WebSocket(signalingUrl);
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  function setStatus(text) {
    if (statusEl) {
      statusEl.textContent = text;
    }
    console.log(text);
  }

  function sendMessage(data) {
    if (signalingSocket.readyState !== WebSocket.OPEN) {
      console.error("WebSocket is not open. Cannot send message:", data);
      return;
    }
    signalingSocket.send(JSON.stringify(data));
  }

  function captureFrame() {
    if (!localVideo.videoWidth || !localVideo.videoHeight) {
      console.warn("Local video not ready for capture");
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
      setStatus("Sent offer");
    } catch (error) {
      console.error(error);
      setStatus(`Camera/WebRTC error: ${error.message || error}`);
    }
  };

  pc.onicecandidate = ({ candidate }) => {
    if (!candidate) {
      return;
    }
    sendMessage({
      type: "candidate",
      candidate: candidate.candidate,
      sdpMid: candidate.sdpMid,
      sdpMLineIndex: candidate.sdpMLineIndex,
    });
  };

  signalingSocket.onmessage = async (message) => {
    const data = JSON.parse(message.data);

    if (data.type === "candidate") {
      if (data.candidate == null) {
        return;
      }
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
        setStatus("Remote answer applied");
      } catch (error) {
        console.error("Error setting remote description from answer", error);
      }
      return;
    }

    if (data.type === "triggerimagecapture") {
      captureFrame();
    }
  };

  pc.ontrack = (event) => {
    if (!remoteVideo.srcObject) {
      remoteVideo.srcObject = event.streams[0];
      setStatus("Receiving remote art stream");
    }
  };

  document.getElementById("captureImage").addEventListener("click", captureFrame);
  setStatus(`Connecting to ${signalingUrl}…`);
})();
