const signalingSocket = new WebSocket("ws://127.0.0.1:8765/signaling");
const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
});

// WebSocket Event Handlers
signalingSocket.onopen = () => {
    console.log("WebSocket connected!");
    document.getElementById("startCall").disabled = false; // Enable button
};

signalingSocket.onerror = (error) => {
    console.error("WebSocket error:", error);
};

signalingSocket.onclose = (event) => {
    console.log("WebSocket closed:", event.code, event.reason);
};

// Capture media stream and add to PeerConnection
async function startCall() {
    const localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    localVideo.srcObject = localStream;
    localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
}

// Unified function for sending WebSocket messages
function sendMessage(data) {
    if (signalingSocket.readyState === WebSocket.OPEN) {
        console.log("Sending message:", data);
        signalingSocket.send(JSON.stringify(data));
    } else {
        console.error("WebSocket is not open. Cannot send message:", data);
    }
}

// Start call when button is clicked
document.getElementById("startCall").onclick = async () => {
    if (signalingSocket.readyState === WebSocket.OPEN) {
        await startCall();
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        sendMessage({ type: "offer", sdp: offer.sdp });
    } else {
        console.error("WebSocket is not open, cannot start call");
    }
};

// Send ICE candidates to the signaling server
pc.onicecandidate = ({ candidate }) => {
    if (candidate) {
        sendMessage({
            type: "candidate",
            candidate: candidate.candidate,
            sdpMid: candidate.sdpMid,
            sdpMLineIndex: candidate.sdpMLineIndex
        });
    }
};

// Handle incoming messages from the signaling server
signalingSocket.onmessage = async (message) => {
    const data = JSON.parse(message.data);

    if (data.type === "candidate") {
        // Handle incoming ICE candidate
        if (data.candidate && (data.sdpMid !== null || data.sdpMLineIndex !== null)) {
            try {
                const candidate = new RTCIceCandidate({
                    candidate: data.candidate,
                    sdpMid: data.sdpMid,
                    sdpMLineIndex: data.sdpMLineIndex
                });
                await pc.addIceCandidate(candidate);
                console.log("Added ICE candidate:", candidate);
            } catch (error) {
                console.error("Error adding received ICE candidate", error);
            }
        } else {
            console.warn("Skipping ICE candidate due to missing sdpMid or sdpMLineIndex", data);
        }

    } else if (data.type === "offer") {
        // Handle incoming SDP offer
        try {
            await pc.setRemoteDescription(new RTCSessionDescription(data));
            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);
            sendMessage({ type: "answer", sdp: answer.sdp });
            console.log("Sent answer to offer");
        } catch (error) {
            console.error("Error handling received offer", error);
        }

    } else if (data.type === "answer") {
        // Handle incoming SDP answer
        try {
            await pc.setRemoteDescription(new RTCSessionDescription(data));
            console.log("Set remote description from answer");
        } catch (error) {
            console.error("Error setting remote description from answer", error);
        }
    }
};

// Handle track event for remote stream
pc.ontrack = (event) => {
    if (!remoteVideo.srcObject) {
        remoteVideo.srcObject = event.streams[0];
    }
};
