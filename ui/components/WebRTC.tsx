"use client";

import { useEffect, useRef, useState } from "react";

export default function WebRTC() {
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const capturedImageRef = useRef<HTMLImageElement>(null);
  const signalingSocketRef = useRef<WebSocket | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  type SignalingMessage =
    | { type: "offer" | "answer"; sdp: string }
    | {
        type: "candidate";
        candidate: string;
        sdpMid?: string;
        sdpMLineIndex?: number;
      }
    | { type: "image"; data: string }
    | { type: "triggerimagecapture" };

  const connectWebSocket = () => {
    const WS_URL = process.env.NEXT_PUBLIC_WS_URL;
    if (!WS_URL) {
      console.error("WebSocket URL not configured");
      return;
    }

    signalingSocketRef.current = new WebSocket(WS_URL);
    setupWebSocketHandlers();
  };

  useEffect(() => {
    pcRef.current = new RTCPeerConnection({
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
        { urls: "stun:stun1.l.google.com:19302" },
      ],
    });

    connectWebSocket();
    setupPeerConnectionHandlers();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      signalingSocketRef.current?.close();
      pcRef.current?.close();
    };
  }, []);

  const setupWebSocketHandlers = () => {
    if (!signalingSocketRef.current) return;

    signalingSocketRef.current.onopen = () => {
      console.log("WebSocket connected!");
      setIsConnected(true);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };

    signalingSocketRef.current.onclose = () => {
      console.log("WebSocket disconnected!");
      setIsConnected(false);

      // Attempt to reconnect after 5 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log("Attempting to reconnect...");
        connectWebSocket();
      }, 5000);
    };

    signalingSocketRef.current.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    signalingSocketRef.current.onmessage = async (message) => {
      try {
        const data = JSON.parse(message.data);
        handleSignalingMessage(data);
      } catch (error) {
        console.error("Error handling message:", error);
      }
    };
  };

  const setupPeerConnectionHandlers = () => {
    if (!pcRef.current) return;

    pcRef.current.onicecandidate = ({ candidate }) => {
      if (candidate) {
        sendMessage({
          type: "candidate",
          candidate: candidate.candidate,
          sdpMid: candidate.sdpMid!,
          sdpMLineIndex: candidate.sdpMLineIndex!,
        });
      }
    };

    pcRef.current.ontrack = (event) => {
      if (remoteVideoRef.current && !remoteVideoRef.current.srcObject) {
        remoteVideoRef.current.srcObject = event.streams[0];
      }
    };
  };

  const sendMessage = (data: SignalingMessage) => {
    if (signalingSocketRef.current?.readyState === WebSocket.OPEN) {
      signalingSocketRef.current.send(JSON.stringify(data));
    }
  };

  const startCall = async () => {
    try {
      const localStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      if (localVideoRef.current) {
        localVideoRef.current.srcObject = localStream;
      }

      localStream.getTracks().forEach((track) => {
        pcRef.current?.addTrack(track, localStream);
      });

      const offer = await pcRef.current?.createOffer();
      await pcRef.current?.setLocalDescription(offer);
      sendMessage({ type: "offer", sdp: offer!.sdp! });
    } catch (error) {
      console.error("Error starting call:", error);
    }
  };

  const captureImage = () => {
    if (!localVideoRef.current) return;

    const canvas = document.createElement("canvas");
    canvas.width = localVideoRef.current.videoWidth;
    canvas.height = localVideoRef.current.videoHeight;

    const context = canvas.getContext("2d");
    if (context && localVideoRef.current) {
      context.drawImage(
        localVideoRef.current,
        0,
        0,
        canvas.width,
        canvas.height
      );
      const imageData = canvas.toDataURL("image/png");

      if (capturedImageRef.current) {
        capturedImageRef.current.src = imageData;
        capturedImageRef.current.style.display = "block";
      }

      sendMessage({ type: "image", data: imageData });
    }
  };

  const handleSignalingMessage = async (data: SignalingMessage) => {
    if (!pcRef.current) return;

    switch (data.type) {
      case "candidate":
        if (
          data.candidate &&
          (data.sdpMid !== null || data.sdpMLineIndex !== null)
        ) {
          await pcRef.current.addIceCandidate(new RTCIceCandidate(data));
        }
        break;
      case "offer":
        await pcRef.current.setRemoteDescription(
          new RTCSessionDescription(data)
        );
        const answer = await pcRef.current.createAnswer();
        await pcRef.current.setLocalDescription(answer);
        sendMessage({ type: "answer", sdp: answer!.sdp! });
        break;
      case "answer":
        await pcRef.current.setRemoteDescription(
          new RTCSessionDescription(data)
        );
        break;
      case "triggerimagecapture":
        captureImage();
        break;
    }
  };

  return (
    <div className="webrtc-container bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-200 p-4 rounded shadow-lg">
      <h2 className="text-xl font-semibold mb-4">WebRTC Video Chat</h2>
      <div className="status mb-4">
        <span
          className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
            isConnected
              ? "bg-green-100 text-green-800"
              : "bg-red-100 text-red-800"
          }`}
        >
          {isConnected ? "Connected" : "Disconnected"}
        </span>
      </div>
      <div className="video-container flex flex-col md:flex-row gap-4">
        <video
          ref={localVideoRef}
          autoPlay
          muted
          playsInline
          className="border rounded w-full md:w-1/2"
        />
        <video
          ref={remoteVideoRef}
          autoPlay
          playsInline
          className="border rounded w-full md:w-1/2"
        />
      </div>
      <div className="controls mt-6 flex gap-4">
        <button
          onClick={startCall}
          disabled={!isConnected}
          className={`px-4 py-2 rounded ${
            isConnected
              ? "bg-blue-500 hover:bg-blue-600 text-white"
              : "bg-gray-400 text-gray-700 cursor-not-allowed"
          }`}
        >
          Start Call
        </button>
        <button
          onClick={captureImage}
          disabled={!isConnected}
          className={`px-4 py-2 rounded ${
            isConnected
              ? "bg-green-500 hover:bg-green-600 text-white"
              : "bg-gray-400 text-gray-700 cursor-not-allowed"
          }`}
        >
          Capture Image
        </button>
      </div>
      <img
        ref={capturedImageRef}
        alt="Captured Frame"
        className="mt-6 border rounded hidden"
      />
    </div>
  );
}
