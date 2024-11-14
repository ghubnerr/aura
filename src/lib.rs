use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use warp::ws::{Message, WebSocket};
use warp::Filter;

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
    ip: String, // Add IP address field
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

// Helper function to share peers between routes
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

    let peers = peers.lock().await; // Await the async Mutex lock
    for (client_id, client) in peers.iter() {
        if client_id != sender_id {
            let mut client = client.lock().await; // Await the async Mutex lock
            if let Err(e) = client.send(Message::text(serialized_message.clone())).await {
                eprintln!("Error sending message to {}: {}", client_id, e);
            }
        }
    }
}

#[pymodule]
fn aura(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SignalingServer>()?;
    Ok(())
}
