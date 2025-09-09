//! # Signaling Server Module
//!
//! This module provides a WebSocket-based signaling server for WebRTC peer connections.
//! It handles client connections, message routing, and image capture functionality.

use crate::SignalingMessage;

use anyhow::Result;
use base64::engine::general_purpose;
use base64::Engine;
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Python;
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use uuid::Uuid;
use warp::ws::{Message, WebSocket};
use warp::Filter;

/// Type alias for managing connected WebSocket peers
pub type Peers = Arc<Mutex<HashMap<String, Arc<Mutex<SplitSink<WebSocket, Message>>>>>>;

/// WebSocket-based signaling server for WebRTC peer connections.
///
/// The SignalingServer manages client connections, handles WebRTC signaling messages,
/// and provides image capture functionality for connected clients.
#[pyclass]
pub struct SignalingServer {
    /// Map of connected client IDs to their WebSocket senders
    pub peers: Peers,
    /// Server port number
    pub port: u16,
    /// Server IP address
    pub ip: String,
    /// Last captured image data from clients
    pub last_captured_image: Arc<Mutex<Option<Vec<u8>>>>,
}

#[pymethods]
impl SignalingServer {
    /// Create a new SignalingServer instance.
    ///
    /// # Arguments
    ///
    /// * `port` - Optional port number (defaults to 3030)
    /// * `ip` - Optional IP address (defaults to "127.0.0.1")
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

    /// Start the signaling server.
    ///
    /// This method starts the WebSocket server and begins accepting client connections.
    #[pyo3(text_signature = "($self)")]
    pub fn start(&self, _py: Python<'_>) -> PyResult<()> {
        let peers = self.peers.clone();
        let port = self.port;
        let ip = self.ip.clone();
        let last_captured_image = self.last_captured_image.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
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

    /// Trigger image capture from connected clients.
    ///
    /// # Arguments
    ///
    /// * `client_id` - Optional specific client ID to capture from. If None, captures from all clients.
    #[pyo3(signature = (client_id=None))]
    #[pyo3(text_signature = "(self, client_id: Optional[str] = None) -> None")]
    pub fn capture(&self, client_id: Option<String>) -> PyResult<()> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async move {
            let peers = peers.lock().await;

            match client_id {
                Some(id) => {
                    if let Some(client) = peers.get(&id) {
                        if let Err(e) = trigger_image_capture(client.clone()).await {
                            eprintln!("Error triggering capture for client {}: {}", id, e);
                        }
                    }
                }
                None => {
                    // Trigger for all clients
                    for (id, client) in peers.iter() {
                        if let Err(e) = trigger_image_capture(client.clone()).await {
                            eprintln!("Error triggering capture for client {}: {}", id, e);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Send a message to a specific connected client.
    ///
    /// # Arguments
    ///
    /// * `client_id` - ID of the target client
    /// * `message` - Message content to send
    ///
    /// # Returns
    ///
    /// True if the message was sent successfully, False otherwise.
    #[pyo3(text_signature = "(self, client_id: str, message: str) -> bool")]
    pub fn send_to_client(&self, client_id: String, message: String) -> PyResult<bool> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let result = rt.block_on(async move {
            if let Some(client) = peers.lock().await.get(&client_id) {
                let mut client = client.lock().await;
                client.send(Message::text(message)).await.is_ok()
            } else {
                false
            }
        });

        Ok(result)
    }

    /// Check if a specific client is currently connected.
    ///
    /// # Arguments
    ///
    /// * `client_id` - ID of the client to check
    ///
    /// # Returns
    ///
    /// True if the client is connected, False otherwise.
    #[pyo3(text_signature = "(self, client_id: str) -> bool")]
    pub fn is_client_connected(&self, client_id: String) -> PyResult<bool> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let is_connected = rt.block_on(async move { peers.lock().await.contains_key(&client_id) });

        Ok(is_connected)
    }

    /// Get a list of all currently connected client IDs.
    ///
    /// # Returns
    ///
    /// A vector of client ID strings.
    #[pyo3(text_signature = "(self) -> List[str]")]
    pub fn get_connected_clients(&self) -> PyResult<Vec<String>> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let clients = rt.block_on(async move { peers.lock().await.keys().cloned().collect() });

        Ok(clients)
    }

    /// Disconnect a specific client from the server.
    ///
    /// # Arguments
    ///
    /// * `client_id` - ID of the client to disconnect
    ///
    /// # Returns
    ///
    /// True if the client was disconnected, False if the client was not found.
    #[pyo3(text_signature = "(self, client_id: str) -> bool")]
    pub fn disconnect_client(&self, client_id: String) -> PyResult<bool> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let was_removed =
            rt.block_on(async move { peers.lock().await.remove(&client_id).is_some() });

        Ok(was_removed)
    }

    /// Check if the server is at capacity (2 or more clients).
    ///
    /// # Returns
    ///
    /// True if the server has reached its capacity limit.
    #[pyo3(text_signature = "(self) -> bool")]
    pub fn is_at_capacity(&self) -> PyResult<bool> {
        Python::with_gil(|py| {
            let count = self.get_client_count(py)?;
            Ok(count >= 2)
        })
    }

    /// Get the current server status information.
    ///
    /// # Returns
    ///
    /// A HashMap containing server status details including IP, port, and client count.
    #[pyo3(text_signature = "(self) -> int")]
    pub fn get_server_status(&self) -> PyResult<HashMap<String, String>> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let mut status = HashMap::new();
        status.insert("ip".to_string(), self.ip.clone());
        status.insert("port".to_string(), self.port.to_string());

        let client_count = rt.block_on(async move { peers.lock().await.len().to_string() });
        status.insert("connected_clients".to_string(), client_count);

        Ok(status)
    }

    /// Broadcast a message to all connected clients.
    ///
    /// # Arguments
    ///
    /// * `message` - Message content to broadcast
    #[pyo3(text_signature = "(self, message: str) -> None")]
    pub fn broadcast_message(&self, message: String, _py: Python<'_>) -> PyResult<()> {
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

    /// Get the current number of connected clients.
    ///
    /// # Returns
    ///
    /// The number of currently connected clients.
    #[pyo3(text_signature = "($self)")]
    pub fn get_client_count(&self, _py: Python<'_>) -> PyResult<usize> {
        let peers = self.peers.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let count = rt.block_on(async move { peers.lock().await.len() });

        Ok(count)
    }

    /// Get the last captured image data from clients.
    ///
    /// # Returns
    ///
    /// The last captured image as bytes, or None if no image has been captured.
    #[pyo3(text_signature = "(self) -> Optional[bytes]")]
    pub fn get_capture<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let last_image = self.last_captured_image.clone();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let image_data = rt.block_on(async move {
            let image = last_image.lock().await;
            image.clone()
        });

        match image_data {
            Some(data) => PyBytes::new_bound_with(py, data.len(), |b| {
                b.copy_from_slice(&data);
                Ok(())
            })
            .map(Some),
            None => Ok(None),
        }
    }
}

/// Warp filter for injecting the peers map into request handlers.
///
/// This function creates a Warp filter that provides access to the shared
/// peers map for WebSocket connection handling.
fn with_peers(
    peers: Peers,
) -> impl Filter<Extract = (Peers,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || peers.clone())
}

/// Handle a new WebSocket connection from a client.
///
/// This function manages the lifecycle of a client connection, including
/// message handling, image capture processing, and cleanup on disconnect.
///
/// # Arguments
///
/// * `ws` - The WebSocket connection
/// * `peers` - Shared map of connected peers
/// * `last_captured_image` - Shared storage for the last captured image
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
                if let Ok(text) = msg.to_str() {
                    println!("Received message from {}: {}", client_id, text);

                    let signaling_message: Result<SignalingMessage, _> = serde_json::from_str(text);
                    match signaling_message {
                        Ok(SignalingMessage::Image { data }) => {
                            println!("Handling image message from client {}", client_id);
                            handle_image_message(data.clone()).await;

                            if let Some(base64_data) = data.split(',').nth(1) {
                                if let Ok(image_bytes) =
                                    general_purpose::STANDARD.decode(base64_data)
                                {
                                    let mut last_image = last_captured_image.lock().await;
                                    *last_image = Some(image_bytes);
                                }
                            }
                        }
                        Ok(message) => {
                            println!("Parsed signaling message: {:?}", message);
                            forward_message(&client_id, &message, &peers).await;
                        }
                        Err(e) => {
                            eprintln!(
                                "Error parsing message from client {}: {} - Error: {:?}",
                                client_id, text, e
                            );
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

/// Process and save an image message from a client.
///
/// This function decodes base64 image data and saves it to a local file.
///
/// # Arguments
///
/// * `data` - Base64-encoded image data with data URL prefix
async fn handle_image_message(data: String) {
    println!("Received image data of length: {}", data.len());

    let base64_data = data.split(',').nth(1).unwrap_or("");
    println!("Base64 content length: {}", base64_data.len());

    match general_purpose::STANDARD.decode(base64_data) {
        Ok(image_bytes) => {
            println!(
                "Decoded image data successfully. Bytes length: {}",
                image_bytes.len()
            );

            if let Err(e) = tokio::fs::write("captured_image.png", &image_bytes).await {
                eprintln!("Failed to save image: {}", e);
            } else {
                println!("Image saved as captured_image.png");
            }
        }
        Err(e) => {
            eprintln!("Failed to decode Base64 image data: {}", e);
        }
    }
}

/// Send an image capture trigger message to a specific client.
///
/// # Arguments
///
/// * `sender` - WebSocket sender for the target client
///
/// # Returns
///
/// Result indicating success or failure of the operation.
async fn trigger_image_capture(
    sender: Arc<Mutex<SplitSink<WebSocket, Message>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let message = serde_json::to_string(&SignalingMessage::TriggerImageCapture)?;
    let mut sender = sender.lock().await;
    sender.send(Message::text(message)).await?;
    println!("Sent image capture trigger to client.");
    Ok(())
}

/// Forward a signaling message to all connected clients except the sender.
///
/// # Arguments
///
/// * `sender_id` - ID of the client who sent the message
/// * `message` - The signaling message to forward
/// * `peers` - Map of all connected peers
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
            let mut client = client.lock().await;
            if let Err(e) = client.send(Message::text(serialized_message.clone())).await {
                eprintln!("Error sending message to {}: {}", client_id, e);
            }
        }
    }
}
