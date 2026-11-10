//! Aura native extension: WebRTC signaling + IVF video streaming for Python.

use pyo3::prelude::*;

mod server;
mod signaling_types;
mod streamer;

pub use signaling_types::SignalingMessage;

use server::SignalingServer;
use streamer::VideoStreamer;

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

    #[test]
    fn test_candidate_accepts_browser_camel_case() {
        let json = r#"{
            "type": "candidate",
            "candidate": "candidate:1 1 UDP 1 1.2.3.4 1234 typ host",
            "sdpMid": "0",
            "sdpMLineIndex": 0
        }"#;
        let parsed: SignalingMessage = serde_json::from_str(json).unwrap();
        match parsed {
            SignalingMessage::Candidate {
                sdp_mid,
                sdp_mline_index,
                ..
            } => {
                assert_eq!(sdp_mid.as_deref(), Some("0"));
                assert_eq!(sdp_mline_index, Some(0));
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }
}
