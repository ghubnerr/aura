use crate::SignalingMessage;

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use pyo3::prelude::*;
use serde_json;
use std::sync::Arc;
use std::{fs::File, io::BufReader, time::Duration};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};
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

#[pyclass]
pub struct VideoStreamer {
    ws_ip: String,
    ws_port: u16,
    ivf_dir: String,
}

#[pymethods]
impl VideoStreamer {
    #[pyo3(text_signature = "(self, ws_ip: str, ws_port: int, ivf_dir: str) -> VideoStreamer")]
    #[new]
    fn new(ws_ip: String, ws_port: u16, ivf_dir: String) -> Self {
        VideoStreamer {
            ws_ip,
            ws_port,
            ivf_dir,
        }
    }

    #[pyo3(text_signature = "(self) -> None")]
    fn start_streaming(&self) -> PyResult<()> {
        let ws_ip = self.ws_ip.clone();
        let ws_port = self.ws_port;
        let ivf_dir = self.ivf_dir.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                if let Err(e) = start_webrtc(&ws_ip, ws_port, &ivf_dir).await {
                    eprintln!("Error starting WebRTC: {}", e);
                }
            });
        });

        Ok(())
    }
}

async fn start_webrtc(ws_ip: &str, ws_port: u16, ivf_dir: &str) -> Result<()> {
    // Create MediaEngine
    let mut m = MediaEngine::default();
    m.register_default_codecs()?;

    // Create a registry for interceptors
    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut m)?;

    // Create the API object
    let api = APIBuilder::new()
        .with_media_engine(m)
        .with_interceptor_registry(registry)
        .build();

    // Prepare the configuration
    let config = RTCConfiguration {
        ice_servers: vec![RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".to_owned()],
            ..Default::default()
        }],
        ..Default::default()
    };

    // Create a new RTCPeerConnection
    let peer_connection = Arc::new(api.new_peer_connection(config).await?);

    // Create video track
    let video_track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_VP8.to_owned(),
            ..Default::default()
        },
        "video".to_owned(),
        "webcam".to_owned(),
    ));

    // Add track to peer connection
    let rtp_sender = peer_connection
        .add_track(Arc::clone(&video_track) as Arc<dyn TrackLocal + Send + Sync>)
        .await?;

    // Handle RTCP packets
    tokio::spawn(async move {
        let mut rtcp_buf = vec![0u8; 1500];
        while let Ok((_, _)) = rtp_sender.read(&mut rtcp_buf).await {}
    });

    // Connect to signaling server
    let (ws_stream, _) = connect_async(format!("ws://{}:{}/signaling", ws_ip, ws_port)).await?;
    let (mut write, mut read) = ws_stream.split();
    let write = Arc::new(Mutex::new(write));
    let pc = Arc::clone(&peer_connection);

    // Handle connection state changes
    peer_connection.on_peer_connection_state_change(Box::new(move |s: RTCPeerConnectionState| {
        println!("Connection State has changed: {s}");
        Box::pin(async {})
    }));

    // Handle incoming messages
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
                                .send(Message::Text(serde_json::to_string(&msg).unwrap()))
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
                        SignalingMessage::Image { data: _ } => {
                            println!("Received image message - ignoring in WebRTC context");
                        }
                        SignalingMessage::TriggerImageCapture => {
                            println!("Received trigger capture message - ignoring in WebRTC context");
                        }
                    }
                }
            }
        }
    });

    println!("Starting video stream...");
    watchand_stream_video(ivf_dir, video_track).await?;

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

//File watcher
async fn watchand_stream_video(directory: &str, track: Arc<TrackLocalStaticSample>) -> Result<()> {
    // Create a channel for file events
    let (tx, mut rx) = mpsc::channel(100);

    // Create an async file watcher
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                tx.blocking_send(event).expect("Failed to send event");
            }
        },
        Config::default(),
    )?;

    // Start watching the specified directory
    watcher.watch(std::path::Path::new(directory), RecursiveMode::NonRecursive)?;

    println!("Watching directory: {}", directory);

    let mut current_file = String::new();

    // Event loop
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