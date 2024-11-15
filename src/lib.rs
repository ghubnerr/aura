use anyhow::Result;
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::{fs::File, io::BufReader, time::Duration};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message as TungsteniteMessage};
use uuid::Uuid;
use warp::ws::{Message, WebSocket};
use warp::Filter;

use webrtc::{
    api::{
        interceptor_registry::register_default_interceptors,
        media_engine::{MediaEngine, MIME_TYPE_VP8},
        APIBuilder,
    },
    ice_transport::{ice_candidate::RTCIceCandidateInit, ice_server::RTCIceServer},
    interceptor::registry::Registry,
    media::{io::ivf_reader::IVFReader, Sample},
    peer_connection::{
        configuration::RTCConfiguration, peer_connection_state::RTCPeerConnectionState,
        sdp::session_description::RTCSessionDescription,
    },
    rtp_transceiver::rtp_codec::RTCRtpCodecCapability,
    track::track_local::{track_local_static_sample::TrackLocalStaticSample, TrackLocal},
};

type Peers = Arc<Mutex<HashMap<String, Arc<Mutex<SplitSink<WebSocket, Message>>>>>>;

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
enum SignalingMessage {
    Offer {
        sdp: String,
    },
    Answer {
        sdp: String,
    },
    Candidate {
        candidate: String,
        sdp_mid: Option<String>,
        sdp_mline_index: Option<u32>,
    },
}
#[pyclass]
struct SignalingServer {
    peers: Peers,
    port: u16,
    ip: String, 
}

#[pymethods]
impl SignalingServer {
    #[new]
    #[pyo3(signature = (port=None, ip=None))]
    fn new(port: Option<u16>, ip: Option<String>) -> Self {
        SignalingServer {
            peers: Arc::new(Mutex::new(HashMap::new())),
            port: port.unwrap_or(3030),
            ip: ip.unwrap_or_else(|| "127.0.0.1".to_string()),
        }
    }

    #[pyo3(text_signature = "($self)")]
    fn start(&self, _py: Python<'_>) -> PyResult<()> {
        let peers = self.peers.clone();
        let port = self.port;
        let ip = self.ip.clone();

        // Run the async runtime in a separate thread
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let signaling_route = warp::path("signaling")
                    .and(warp::ws())
                    .and(with_peers(peers.clone()))
                    .map(|ws: warp::ws::Ws, peers| {
                        ws.on_upgrade(move |socket| handle_connection(socket, peers))
                    });

                // Parse IP address from string
                let ip_addr: std::net::IpAddr = ip.parse().expect("Invalid IP address");
                println!("Signaling server running on ws://{}:{}/signaling", ip, port);
                warp::serve(signaling_route).run((ip_addr, port)).await;
            });
        });

        Ok(())
    }

    /// Send a message to all connected peers
    #[pyo3(text_signature = "($self, message)")]
    fn broadcast_message(&self, message: String, _py: Python<'_>) -> PyResult<()> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async move {
            let peers = peers.lock().await;
            for (_, client) in peers.iter() {
                let mut client = client.lock().await;
                if let Err(e) = client.send(Message::text(message.clone())).await {
                    eprintln!("Error broadcasting message: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Get the number of connected clients
    #[pyo3(text_signature = "($self)")]
    fn get_client_count(&self, _py: Python<'_>) -> PyResult<usize> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let count = rt.block_on(async move { peers.lock().await.len() });

        Ok(count)
    }
}

fn with_peers(
    peers: Peers,
) -> impl Filter<Extract = (Peers,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || peers.clone())
}

async fn handle_connection(ws: WebSocket, peers: Peers) {
    let (sender, mut receiver) = ws.split();
    let sender = Arc::new(Mutex::new(sender));

    let client_id = Uuid::new_v4().to_string();
    peers.lock().await.insert(client_id.clone(), sender.clone());

    println!("Client {} connected", client_id);

    while let Some(result) = receiver.next().await {
        match result {
            Ok(msg) => {
                if let Ok(text) = msg.to_str() {
                    println!("Received message from {}: {}", client_id, text);

                    // Attempt to parse the message
                    let signaling_message: Result<SignalingMessage, _> = serde_json::from_str(text);
                    match signaling_message {
                        Ok(message) => {
                            println!("Parsed message successfully: {:?}", message);
                            forward_message(&client_id, &message, &peers).await;
                        }
                        Err(e) => {
                            eprintln!(
                                "Error parsing message from client {}: {} - Error: {:?}",
                                client_id, text, e
                            );
                            // Log and continue, but do not break the connection
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Error receiving message for client {}: {}", client_id, e);
                break;
            }
        }
    }

    peers.lock().await.remove(&client_id);
    println!("Client {} disconnected", client_id);
}

async fn forward_message(sender_id: &str, message: &SignalingMessage, peers: &Peers) {
    let serialized_message = match serde_json::to_string(message) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("Failed to serialize message: {}", e);
            return;
        }
    };

    let peers = peers.lock().await; 
    for (client_id, client) in peers.iter() {
        if client_id != sender_id {
            let mut client = client.lock().await; // Await the async Mutex lock
            if let Err(e) = client.send(Message::text(serialized_message.clone())).await {
                eprintln!("Error sending message to {}: {}", client_id, e);
            }
        }
    }
}

#[pyclass]
pub struct VideoStreamer {
    ws_ip: String,
    ws_port: u16,
    ivf_dir: String,
}

#[pymethods]
impl VideoStreamer {
    #[new]
    fn new(ws_ip: String, ws_port: u16, ivf_dir: String) -> Self {
        VideoStreamer {
            ws_ip,
            ws_port,
            ivf_dir,
        }
    }

    fn start_streaming(&self) -> PyResult<()> {
        let ws_ip = self.ws_ip.clone();
        let ws_port = self.ws_port;
        let ivf_dir = self.ivf_dir.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                // Setup WebRTC and signaling
                if let Err(e) = start_webrtc(&ws_ip, ws_port, &ivf_dir).await {
                    eprintln!("Error starting WebRTC: {}", e);
                }
            });
        });

        Ok(())
    }
}

async fn start_webrtc(ws_ip: &str, ws_port: u16, ivf_dir: &str) -> Result<()> {
    let mut m = MediaEngine::default();
    m.register_default_codecs()?;

    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut m)?;

    let api = APIBuilder::new()
        .with_media_engine(m)
        .with_interceptor_registry(registry)
        .build();

    let config = RTCConfiguration {
        ice_servers: vec![RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".to_owned()],
            ..Default::default()
        }],
        ..Default::default()
    };

    let peer_connection = Arc::new(api.new_peer_connection(config).await?);

    let video_track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_VP8.to_owned(),
            ..Default::default()
        },
        "video".to_owned(),
        "webcam".to_owned(),
    ));

    let rtp_sender = peer_connection
        .add_track(Arc::clone(&video_track) as Arc<dyn TrackLocal + Send + Sync>)
        .await?;

    tokio::spawn(async move {
        let mut rtcp_buf = vec![0u8; 1500];
        while let Ok((_, _)) = rtp_sender.read(&mut rtcp_buf).await {}
    });

    // Connect to signaling server
    let (ws_stream, _) = connect_async(format!("ws://{}:{}/signaling", ws_ip, ws_port)).await?;
    let (mut write, mut read) = ws_stream.split();
    let write = Arc::new(Mutex::new(write));
    let pc = Arc::clone(&peer_connection);

    peer_connection.on_peer_connection_state_change(Box::new(move |s: RTCPeerConnectionState| {
        println!("Connection State has changed: {s}");
        Box::pin(async {})
    }));

    let write_clone = Arc::clone(&write);
    tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            if let Ok(msg) = msg {
                let text = msg.to_string();
                if let Ok(signal) = serde_json::from_str::<SignalingMessage>(&text) {
                    match signal {
                        SignalingMessage::Offer { sdp } => {
                            let offer = RTCSessionDescription::offer(sdp).unwrap();
                            pc.set_remote_description(offer).await.unwrap();

                            let answer = pc.create_answer(None).await.unwrap();
                            pc.set_local_description(answer.clone()).await.unwrap();

                            let msg = SignalingMessage::Answer { sdp: answer.sdp };
                            let mut write = write_clone.lock().await;
                            write
                                .send(TungsteniteMessage::Text(
                                    serde_json::to_string(&msg).unwrap(),
                                ))
                                .await
                                .unwrap();
                        }
                        SignalingMessage::Answer { sdp } => {
                            let answer = RTCSessionDescription::answer(sdp).unwrap();
                            pc.set_remote_description(answer).await.unwrap();
                        }
                        SignalingMessage::Candidate {
                            candidate,
                            sdp_mid,
                            sdp_mline_index,
                        } => {
                            let candidate = RTCIceCandidateInit {
                                candidate,
                                sdp_mid,
                                sdp_mline_index: sdp_mline_index.map(|x| x as u16),
                                username_fragment: None,
                            };
                            if let Err(e) = pc.add_ice_candidate(candidate).await {
                                println!("Error adding ICE candidate: {}", e);
                            }
                        }
                    }
                }
            }
        }
    });

    println!("Starting video stream...");
    watch_and_stream_video(ivf_dir, video_track).await?;

    Ok(())
}

async fn write_video_to_track(path: &str, track: Arc<TrackLocalStaticSample>) -> Result<()> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let (mut ivf, header) = IVFReader::new(reader)?;

    let sleep_time = Duration::from_millis(
        ((1000 * header.timebase_numerator) / header.timebase_denominator) as u64,
    );
    let mut ticker = tokio::time::interval(sleep_time);

    loop {
        let frame = ivf.parse_next_frame()?.0;
        track
            .write_sample(&Sample {
                data: frame.freeze(),
                duration: Duration::from_secs(1),
                ..Default::default()
            })
            .await?;
        ticker.tick().await;
    }
}

async fn watch_and_stream_video(directory: &str, track: Arc<TrackLocalStaticSample>) -> Result<()> {
    let (tx, mut rx) = mpsc::channel(100);

    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                tx.blocking_send(event).expect("Failed to send event");
            }
        },
        Config::default(),
    )?;

    watcher.watch(std::path::Path::new(directory), RecursiveMode::NonRecursive)?;

    println!("Watching directory: {}", directory);

    let mut current_file;

    while let Some(event) = rx.recv().await {
        if let Event {
            kind: EventKind::Modify(_),
            paths,
            ..
        } = event
        {
            for path in paths {
                if let Some(ext) = path.extension() {
                    if ext == "ivf" {
                        println!("Detected change in file: {:?}", path);
                        current_file = path.to_string_lossy().to_string();
                        if let Err(e) =
                            write_video_to_track(&current_file, Arc::clone(&track)).await
                        {
                            println!("Error streaming video: {}", e);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[pymodule]
fn aura(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SignalingServer>()?;
    m.add_class::<VideoStreamer>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = SignalingServer::new(Some(3031), Some("127.0.0.1".to_string()));
        assert_eq!(server.port, 3031);
        assert_eq!(server.ip, "127.0.0.1");
    }

    #[test] 
    fn test_server_default_values() {
        let server = SignalingServer::new(None, None);
        assert_eq!(server.port, 3030);
        assert_eq!(server.ip, "127.0.0.1");
    }

    #[test]
    fn test_message_serialization() {
        let offer = SignalingMessage::Offer {
            sdp: "test_sdp".to_string(),
        };
        let serialized = serde_json::to_string(&offer).unwrap();
        assert!(serialized.contains("offer"));
        assert!(serialized.contains("test_sdp"));
    }
}
