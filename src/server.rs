//! WebSocket signaling server used by the browser UI and the video streamer.

use crate::SignalingMessage;

use anyhow::Result;
use base64::engine::general_purpose;
use base64::Engine;
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Python;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use uuid::Uuid;
use warp::ws::{Message, WebSocket};
use warp::Filter;

/// Shared map of connected signaling peers.
pub type Peers = Arc<Mutex<HashMap<String, Arc<Mutex<SplitSink<WebSocket, Message>>>>>>;

/// Maximum peers for a single Aura session (browser + streamer).
const MAX_PEERS: usize = 2;

fn shared_runtime() -> &'static Runtime {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("failed to create Tokio runtime"))
}

/// Python-facing WebSocket signaling server.
#[pyclass]
pub struct SignalingServer {
    pub peers: Peers,
    pub port: u16,
    pub ip: String,
    pub last_captured_image: Arc<Mutex<Option<Vec<u8>>>>,
}

#[pymethods]
impl SignalingServer {
    #[new]
    #[pyo3(signature = (port=None, ip=None))]
    #[pyo3(
        text_signature = "(self, port: Optional[int] = None, ip: Optional[str] = None) -> SignalingServer"
    )]
    pub fn new(port: Option<u16>, ip: Option<String>) -> Self {
        SignalingServer {
            peers: Arc::new(Mutex::new(HashMap::new())),
            port: port.unwrap_or(3030),
            ip: ip.unwrap_or_else(|| "127.0.0.1".to_string()),
            last_captured_image: Arc::new(Mutex::new(None)),
        }
    }

    /// Start the Warp WebSocket server on a background thread.
    #[pyo3(text_signature = "($self)")]
    pub fn start(&self, _py: Python<'_>) -> PyResult<()> {
        let peers = self.peers.clone();
        let port = self.port;
        let ip = self.ip.clone();
        let last_captured_image = self.last_captured_image.clone();

        std::thread::spawn(move || {
            let rt = Runtime::new().expect("failed to create signaling runtime");
            rt.block_on(async move {
                let signaling_route = warp::path("signaling")
                    .and(warp::ws())
                    .and(with_peers(peers.clone()))
                    .and(warp::any().map(move || last_captured_image.clone()))
                    .map(|ws: warp::ws::Ws, peers, last_image| {
                        ws.on_upgrade(move |socket| handle_connection(socket, peers, last_image))
                    });

                let ip_addr: std::net::IpAddr = ip.parse().expect("Invalid IP address");
                println!("Signaling server running on ws://{}:{}/signaling", ip, port);
                warp::serve(signaling_route).run((ip_addr, port)).await;
            });
        });

        Ok(())
    }

    /// Ask one client (or all clients) to capture a webcam frame.
    #[pyo3(signature = (client_id=None))]
    #[pyo3(text_signature = "(self, client_id: Optional[str] = None) -> None")]
    pub fn capture(&self, client_id: Option<String>) -> PyResult<()> {
        let peers = self.peers.clone();
        shared_runtime().block_on(async move {
            let peers = peers.lock().await;
            match client_id {
                Some(id) => {
                    if let Some(client) = peers.get(&id) {
                        if let Err(err) = trigger_image_capture(client.clone()).await {
                            eprintln!("Error triggering capture for client {}: {}", id, err);
                        }
                    }
                }
                None => {
                    for (id, client) in peers.iter() {
                        if let Err(err) = trigger_image_capture(client.clone()).await {
                            eprintln!("Error triggering capture for client {}: {}", id, err);
                        }
                    }
                }
            }
        });
        Ok(())
    }

    #[pyo3(text_signature = "(self, client_id: str, message: str) -> bool")]
    pub fn send_to_client(&self, client_id: String, message: String) -> PyResult<bool> {
        let peers = self.peers.clone();
        let result = shared_runtime().block_on(async move {
            if let Some(client) = peers.lock().await.get(&client_id) {
                let mut client = client.lock().await;
                client.send(Message::text(message)).await.is_ok()
            } else {
                false
            }
        });
        Ok(result)
    }

    #[pyo3(text_signature = "(self, client_id: str) -> bool")]
    pub fn is_client_connected(&self, client_id: String) -> PyResult<bool> {
        let peers = self.peers.clone();
        let is_connected =
            shared_runtime().block_on(async move { peers.lock().await.contains_key(&client_id) });
        Ok(is_connected)
    }

    #[pyo3(text_signature = "(self) -> List[str]")]
    pub fn get_connected_clients(&self) -> PyResult<Vec<String>> {
        let peers = self.peers.clone();
        let clients = shared_runtime()
            .block_on(async move { peers.lock().await.keys().cloned().collect() });
        Ok(clients)
    }

    #[pyo3(text_signature = "(self, client_id: str) -> bool")]
    pub fn disconnect_client(&self, client_id: String) -> PyResult<bool> {
        let peers = self.peers.clone();
        let was_removed = shared_runtime()
            .block_on(async move { peers.lock().await.remove(&client_id).is_some() });
        Ok(was_removed)
    }

    #[pyo3(text_signature = "(self) -> bool")]
    pub fn is_at_capacity(&self) -> PyResult<bool> {
        Python::with_gil(|py| {
            let count = self.get_client_count(py)?;
            Ok(count >= MAX_PEERS)
        })
    }

    #[pyo3(text_signature = "(self) -> int")]
    pub fn get_server_status(&self) -> PyResult<HashMap<String, String>> {
        let peers = self.peers.clone();
        let mut status = HashMap::new();
        status.insert("ip".to_string(), self.ip.clone());
        status.insert("port".to_string(), self.port.to_string());

        let client_count =
            shared_runtime().block_on(async move { peers.lock().await.len().to_string() });
        status.insert("connected_clients".to_string(), client_count);
        Ok(status)
    }

    #[pyo3(text_signature = "(self, message: str) -> None")]
    pub fn broadcast_message(&self, message: String, _py: Python<'_>) -> PyResult<()> {
        let peers = self.peers.clone();
        shared_runtime().block_on(async move {
            let peers = peers.lock().await;
            for (_, client) in peers.iter() {
                let mut client = client.lock().await;
                if let Err(err) = client.send(Message::text(message.clone())).await {
                    eprintln!("Error broadcasting message: {}", err);
                }
            }
        });
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_client_count(&self, _py: Python<'_>) -> PyResult<usize> {
        let peers = self.peers.clone();
        let count = shared_runtime().block_on(async move { peers.lock().await.len() });
        Ok(count)
    }

    /// Return the most recently captured webcam PNG bytes, if any.
    #[pyo3(text_signature = "(self) -> Optional[bytes]")]
    pub fn get_capture<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let last_image = self.last_captured_image.clone();
        let image_data =
            shared_runtime().block_on(async move { last_image.lock().await.clone() });

        match image_data {
            Some(data) => PyBytes::new_bound_with(py, data.len(), |buf| {
                buf.copy_from_slice(&data);
                Ok(())
            })
            .map(Some),
            None => Ok(None),
        }
    }
}

fn with_peers(
    peers: Peers,
) -> impl Filter<Extract = (Peers,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || peers.clone())
}

async fn handle_connection(
    ws: WebSocket,
    peers: Peers,
    last_captured_image: Arc<Mutex<Option<Vec<u8>>>>,
) {
    let (sender, mut receiver) = ws.split();
    let sender = Arc::new(Mutex::new(sender));

    let client_id = Uuid::new_v4().to_string();
    peers.lock().await.insert(client_id.clone(), sender.clone());
    println!("Client {} connected", client_id);

    while let Some(result) = receiver.next().await {
        match result {
            Ok(msg) => {
                let Ok(text) = msg.to_str() else {
                    continue;
                };

                match serde_json::from_str::<SignalingMessage>(text) {
                    Ok(SignalingMessage::Image { data }) => {
                        if let Some(bytes) = decode_data_url_png(&data) {
                            let mut last_image = last_captured_image.lock().await;
                            *last_image = Some(bytes.clone());
                            if let Err(err) = tokio::fs::write("captured_image.png", &bytes).await {
                                eprintln!("Failed to save image: {}", err);
                            }
                        }
                    }
                    Ok(message) => {
                        forward_message(&client_id, &message, &peers).await;
                    }
                    Err(err) => {
                        eprintln!(
                            "Error parsing message from client {}: {} ({:?})",
                            client_id, text, err
                        );
                    }
                }
            }
            Err(err) => {
                eprintln!("Error receiving message for client {}: {}", client_id, err);
                break;
            }
        }
    }

    peers.lock().await.remove(&client_id);
    println!("Client {} disconnected", client_id);
}

fn decode_data_url_png(data: &str) -> Option<Vec<u8>> {
    let base64_data = data.split(',').nth(1)?;
    general_purpose::STANDARD.decode(base64_data).ok()
}

async fn trigger_image_capture(
    sender: Arc<Mutex<SplitSink<WebSocket, Message>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let message = serde_json::to_string(&SignalingMessage::TriggerImageCapture)?;
    let mut sender = sender.lock().await;
    sender.send(Message::text(message)).await?;
    Ok(())
}

async fn forward_message(sender_id: &str, message: &SignalingMessage, peers: &Peers) {
    let Ok(serialized) = serde_json::to_string(message) else {
        eprintln!("Failed to serialize signaling message");
        return;
    };

    let peers = peers.lock().await;
    for (client_id, client) in peers.iter() {
        if client_id == sender_id {
            continue;
        }
        let mut client = client.lock().await;
        if let Err(err) = client.send(Message::text(serialized.clone())).await {
            eprintln!("Error sending message to {}: {}", client_id, err);
        }
    }
}
