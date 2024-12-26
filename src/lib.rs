use pyo3::prelude::*;

mod queue;
mod server;
mod signaling_types;
pub use signaling_types::SignalingMessage;

mod streamer;

use queue::QueueManager;
use server::SignalingServer;
use streamer::VideoStreamer;

#[pymodule]
fn aura(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SignalingServer>()?;
    m.add_class::<VideoStreamer>()?;
    m.add_class::<QueueManager>()?;
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
