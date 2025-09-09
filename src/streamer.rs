//! # Video Streamer Module
//!
//! This module provides WebRTC-based video streaming capabilities with support for
//! IVF (Indeo Video Format) files and real-time file watching.

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
        sdp::session_description::RTCSessionDescription, RTCPeerConnection,
    },
    rtp_transceiver::rtp_codec::RTCRtpCodecCapability,
    track::track_local::{track_local_static_sample::TrackLocalStaticSample, TrackLocal},
};

/// WebRTC video streamer for real-time video transmission.
///
/// The VideoStreamer handles WebRTC peer connections and streams video content
/// from IVF files to connected clients through a signaling server.
#[pyclass]
pub struct VideoStreamer {
    /// WebSocket server IP address for signaling
    ws_ip: String,
    /// WebSocket server port for signaling
    ws_port: u16,
    /// Directory containing IVF video files to stream
    ivf_dir: String,
    /// Shared reference to the active WebRTC peer connection
    peer_connection: Arc<Mutex<Option<Arc<RTCPeerConnection>>>>,
}

#[pymethods]
impl VideoStreamer {
    /// Create a new VideoStreamer instance.
    ///
    /// # Arguments
    ///
    /// * `ws_ip` - IP address of the WebSocket signaling server
    /// * `ws_port` - Port number of the WebSocket signaling server
    /// * `ivf_dir` - Directory path containing IVF video files to stream
    #[pyo3(text_signature = "(self, ws_ip: str, ws_port: int, ivf_dir: str) -> VideoStreamer")]
    #[new]
    fn new(ws_ip: String, ws_port: u16, ivf_dir: String) -> Self {
        VideoStreamer {
            ws_ip,
            ws_port,
            ivf_dir,
            peer_connection: Arc::new(Mutex::new(None)),
        }
    }

    /// Start the video streaming process.
    ///
    /// This method initiates the WebRTC connection and begins streaming
    /// video content from the specified IVF directory.
    #[pyo3(text_signature = "(self) -> None")]
    fn start_streaming(&self) -> PyResult<()> {
        let ws_ip = self.ws_ip.clone();
        let ws_port = self.ws_port;
        let ivf_dir = self.ivf_dir.clone();
        let peer_connection_store = self.peer_connection.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                if let Err(e) = start_webrtc(&ws_ip, ws_port, &ivf_dir, peer_connection_store).await
                {
                    eprintln!("Error starting WebRTC: {}", e);
                }
            });
        });

        Ok(())
    }

    /// Get WebRTC connection statistics.
    ///
    /// Returns a JSON string containing detailed statistics about the
    /// current WebRTC peer connection, including bandwidth, latency, and packet loss.
    #[pyo3(text_signature = "(self) -> str")]
    fn get_stats(&self) -> PyResult<String> {
        let peer_connection = self.peer_connection.clone();

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            if let Some(pc) = peer_connection.lock().await.as_ref() {
                let stats = pc.get_stats().await;
                match serde_json::to_string(&stats) {
                    Ok(stats_string) => Ok(stats_string),
                    Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to serialize stats: {}",
                        e
                    ))),
                }
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "No active peer connection",
                ))
            }
        })
    }

    /// Close the active WebRTC peer connection.
    ///
    /// This method gracefully closes the current peer connection and
    /// cleans up associated resources.
    #[pyo3(text_signature = "(self) -> None")]
    fn close_connection(&self) -> PyResult<()> {
        let peer_connection = self.peer_connection.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            if let Some(pc) = peer_connection.lock().await.as_ref() {
                if let Err(e) = pc.close().await {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to close connection: {}",
                        e
                    )));
                }
                *peer_connection.lock().await = None;
                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "No active peer connection",
                ))
            }
        })
    }
}

/// Initialize and start the WebRTC streaming session.
///
/// This function sets up the WebRTC peer connection, establishes signaling
/// communication, and begins streaming video content from IVF files.
///
/// # Arguments
///
/// * `ws_ip` - WebSocket server IP address
/// * `ws_port` - WebSocket server port
/// * `ivf_dir` - Directory containing IVF video files
/// * `peer_connection_store` - Shared storage for the peer connection

async fn start_webrtc(
    ws_ip: &str,
    ws_port: u16,
    ivf_dir: &str,
    peer_connection_store: Arc<Mutex<Option<Arc<RTCPeerConnection>>>>,
) -> Result<()> {
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
    *peer_connection_store.lock().await = Some(Arc::clone(&peer_connection));

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

    let (ws_stream, _) = connect_async(format!("ws://{}:{}/signaling", ws_ip, ws_port)).await?;
    let (write, mut read) = ws_stream.split();
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
                            println!(
                                "Received trigger capture message - ignoring in WebRTC context"
                            );
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

/// Write video frames from an IVF file to a WebRTC track.
///
/// This function reads an IVF file frame by frame and writes the video data
/// to the specified WebRTC track at the appropriate frame rate.
///
/// # Arguments
///
/// * `path` - Path to the IVF video file
/// * `track` - WebRTC track to write video data to
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

/// Watch a directory for IVF file changes and stream video content.
///
/// This function monitors the specified directory for new or modified IVF files
/// and automatically streams them to the WebRTC track when detected.
///
/// # Arguments
///
/// * `directory` - Directory path to watch for IVF files
/// * `track` - WebRTC track to stream video data to
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
                        let current_file = path.to_string_lossy().to_string();
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
