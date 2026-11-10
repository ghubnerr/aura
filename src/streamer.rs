//! WebRTC peer that answers browser offers and streams IVF/VP8 art frames.

use crate::SignalingMessage;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use pyo3::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_VP8};
use webrtc::api::APIBuilder;
use webrtc::ice_transport::ice_candidate::RTCIceCandidateInit;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::media::io::ivf_reader::IVFReader;
use webrtc::media::Sample;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use webrtc::track::track_local::TrackLocal;

type WsWrite = Arc<
    Mutex<
        futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            Message,
        >,
    >,
>;

/// Python-facing IVF → WebRTC streamer.
#[pyclass]
pub struct VideoStreamer {
    ws_ip: String,
    ws_port: u16,
    ivf_dir: String,
    active: Arc<AtomicBool>,
}

#[pymethods]
impl VideoStreamer {
    #[new]
    #[pyo3(text_signature = "(self, ws_ip: str, ws_port: int, ivf_dir: str) -> VideoStreamer")]
    fn new(ws_ip: String, ws_port: u16, ivf_dir: String) -> Self {
        VideoStreamer {
            ws_ip,
            ws_port,
            ivf_dir,
            active: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Connect to the signaling server and begin watching `ivf_dir` for VP8 files.
    #[pyo3(text_signature = "(self) -> None")]
    fn start_streaming(&self) -> PyResult<()> {
        if self.active.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        let ws_ip = self.ws_ip.clone();
        let ws_port = self.ws_port;
        let ivf_dir = self.ivf_dir.clone();
        let active = self.active.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("failed to create streamer runtime");
            rt.block_on(async move {
                if let Err(err) = start_webrtc(&ws_ip, ws_port, &ivf_dir).await {
                    eprintln!("Error starting WebRTC: {}", err);
                }
                active.store(false, Ordering::SeqCst);
            });
        });

        Ok(())
    }

    /// Placeholder until frame grab from the outbound track is implemented.
    #[pyo3(text_signature = "(self) -> None")]
    fn take_screenshot(&self) -> PyResult<()> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No active peer connection",
            ));
        }
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Screenshot capture is not implemented for the IVF streamer yet",
        ))
    }

    /// Placeholder stats API kept for Python tests / future UI.
    #[pyo3(text_signature = "(self) -> dict")]
    fn get_stats(&self) -> PyResult<std::collections::HashMap<String, String>> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No active peer connection",
            ));
        }
        let mut stats = std::collections::HashMap::new();
        stats.insert("ws_ip".to_string(), self.ws_ip.clone());
        stats.insert("ws_port".to_string(), self.ws_port.to_string());
        stats.insert("ivf_dir".to_string(), self.ivf_dir.clone());
        stats.insert("active".to_string(), "true".to_string());
        Ok(stats)
    }

    /// Mark the streamer inactive. The background WebRTC task exits on its own
    /// when the signaling socket closes or the process ends.
    #[pyo3(text_signature = "(self) -> None")]
    fn close_connection(&self) -> PyResult<()> {
        if !self.active.swap(false, Ordering::SeqCst) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No active peer connection",
            ));
        }
        Ok(())
    }
}

async fn start_webrtc(ws_ip: &str, ws_port: u16, ivf_dir: &str) -> Result<()> {
    let mut media_engine = MediaEngine::default();
    media_engine.register_default_codecs()?;

    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut media_engine)?;

    let api = APIBuilder::new()
        .with_media_engine(media_engine)
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
        "aura".to_owned(),
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

    peer_connection.on_peer_connection_state_change(Box::new(
        move |state: RTCPeerConnectionState| {
            println!("Connection state changed: {state}");
            Box::pin(async {})
        },
    ));

    // Trickle ICE back to the browser — without this, remote video often stays
    // blank when the peers are not on the exact same host.
    let write_ice = Arc::clone(&write);
    peer_connection.on_ice_candidate(Box::new(move |candidate| {
        let write_ice = Arc::clone(&write_ice);
        Box::pin(async move {
            let Some(candidate) = candidate else {
                return;
            };
            let Ok(init) = candidate.to_json() else {
                return;
            };
            let msg = SignalingMessage::Candidate {
                candidate: init.candidate,
                sdp_mid: init.sdp_mid,
                sdp_mline_index: init.sdp_mline_index.map(|idx| idx as u32),
            };
            if let Ok(text) = serde_json::to_string(&msg) {
                let mut write = write_ice.lock().await;
                let _ = write.send(Message::Text(text)).await;
            }
        })
    }));

    let write_clone = Arc::clone(&write);
    tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            let Ok(msg) = msg else {
                continue;
            };
            let text = msg.to_string();
            let Ok(signal) = serde_json::from_str::<SignalingMessage>(&text) else {
                continue;
            };

            if let Err(err) = handle_signal(signal, &pc, &write_clone).await {
                eprintln!("Signaling handler error: {}", err);
            }
        }
    });

    println!("Starting video stream from {}", ivf_dir);
    watch_and_stream_video(ivf_dir, video_track).await
}

async fn handle_signal(
    signal: SignalingMessage,
    pc: &Arc<webrtc::peer_connection::RTCPeerConnection>,
    write: &WsWrite,
) -> Result<()> {
    match signal {
        SignalingMessage::Offer { sdp } => {
            let offer = RTCSessionDescription::offer(sdp)?;
            pc.set_remote_description(offer).await?;
            let answer = pc.create_answer(None).await?;
            pc.set_local_description(answer.clone()).await?;
            let msg = SignalingMessage::Answer { sdp: answer.sdp };
            let mut write = write.lock().await;
            write
                .send(Message::Text(serde_json::to_string(&msg)?))
                .await?;
        }
        SignalingMessage::Answer { sdp } => {
            let answer = RTCSessionDescription::answer(sdp)?;
            pc.set_remote_description(answer).await?;
        }
        SignalingMessage::Candidate {
            candidate,
            sdp_mid,
            sdp_mline_index,
        } => {
            let candidate = RTCIceCandidateInit {
                candidate,
                sdp_mid,
                sdp_mline_index: sdp_mline_index.map(|idx| idx as u16),
                username_fragment: None,
            };
            pc.add_ice_candidate(candidate).await?;
        }
        SignalingMessage::Image { .. } | SignalingMessage::TriggerImageCapture => {}
    }
    Ok(())
}

/// Load all VP8 frames from an IVF file into memory so the file handle can be
/// released (Windows cannot replace a file the streamer still holds open).
fn load_ivf_frames(path: &Path) -> Result<(Vec<Bytes>, Duration)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let (mut ivf, header) = IVFReader::new(reader)?;

    if header.timebase_denominator == 0 {
        return Err(anyhow!("invalid IVF timebase in {}", path.display()));
    }

    let frame_duration = Duration::from_millis(
        ((1000 * header.timebase_numerator) / header.timebase_denominator).max(1) as u64,
    );

    let mut frames = Vec::new();
    loop {
        match ivf.parse_next_frame() {
            Ok((frame, _)) => frames.push(frame.freeze()),
            Err(_) => break,
        }
    }

    if frames.is_empty() {
        return Err(anyhow!("no frames in {}", path.display()));
    }

    Ok((frames, frame_duration))
}

/// Loop an IVF clip forever until the caller cancels via `tokio::select!`.
async fn loop_ivf_to_track(path: &Path, track: Arc<TrackLocalStaticSample>) -> Result<()> {
    let (frames, frame_duration) = load_ivf_frames(path)?;
    println!(
        "Streaming {} ({} frames, {:?}/frame) — looping until replaced",
        path.display(),
        frames.len(),
        frame_duration
    );

    let mut ticker = tokio::time::interval(frame_duration);
    // Skip the immediate first tick so we don't burst two frames at start.
    ticker.tick().await;

    loop {
        for frame in &frames {
            track
                .write_sample(&Sample {
                    data: frame.clone(),
                    duration: frame_duration,
                    ..Default::default()
                })
                .await?;
            ticker.tick().await;
        }
    }
}

fn pick_ivf(directory: &Path) -> Option<PathBuf> {
    let live = directory.join("live.ivf");
    if live.is_file() {
        return Some(live);
    }

    let mut candidates: Vec<PathBuf> = std::fs::read_dir(directory)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("ivf"))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| !name.starts_with('.'))
                .unwrap_or(false)
        })
        .collect();

    candidates.sort();
    candidates.pop()
}

fn path_from_event(event: &Event) -> Option<PathBuf> {
    match event.kind {
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Any => {}
        // Windows replace often surfaces as Remove + Create / Rename.
        EventKind::Remove(_) => return None,
        _ => {
            // Some backends report Rename under Other / Access — still inspect paths.
        }
    }

    event.paths.iter().find_map(|path| {
        if path.extension().and_then(|ext| ext.to_str()) != Some("ivf") {
            return None;
        }
        let name = path.file_name()?.to_str()?;
        if name.starts_with('.') || name.ends_with(".tmp") {
            return None;
        }
        if path.is_file() {
            Some(path.clone())
        } else {
            None
        }
    })
}

async fn watch_and_stream_video(directory: &str, track: Arc<TrackLocalStaticSample>) -> Result<()> {
    let dir = PathBuf::from(directory);
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }

    let (tx, mut rx) = mpsc::channel::<PathBuf>(16);
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                if let Some(path) = path_from_event(&event) {
                    let _ = tx.blocking_send(path);
                }
            }
        },
        Config::default(),
    )?;

    watcher.watch(Path::new(directory), RecursiveMode::NonRecursive)?;
    println!("Watching directory: {}", directory);

    let mut current = pick_ivf(&dir);
    if let Some(ref path) = current {
        println!("Seeding stream from existing file: {}", path.display());
    } else {
        println!("No IVF yet in {} — waiting for publish", directory);
    }

    loop {
        match current.clone() {
            Some(path) => {
                tokio::select! {
                    result = loop_ivf_to_track(&path, Arc::clone(&track)) => {
                        if let Err(err) = result {
                            eprintln!("Error streaming {}: {}", path.display(), err);
                            // Wait for a replacement rather than spinning on a bad file.
                            if let Some(next) = rx.recv().await {
                                println!("Detected change in file: {:?}", next);
                                current = Some(next);
                            } else {
                                break;
                            }
                        }
                    }
                    next = rx.recv() => {
                        let Some(mut next) = next else { break; };
                        println!("Detected change in file: {:?}", next);
                        // Debounce bursty Windows FS events; always restart so a
                        // same-path replace (live.ivf) reloads emotion frames.
                        while let Ok(more) = rx.try_recv() {
                            println!("Detected change in file: {:?}", more);
                            next = more;
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        current = Some(next);
                    }
                }
            }
            None => {
                if let Some(next) = rx.recv().await {
                    println!("Detected new IVF: {:?}", next);
                    current = Some(next);
                } else {
                    break;
                }
            }
        }
    }

    Ok(())
}
