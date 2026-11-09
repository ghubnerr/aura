//! JSON signaling messages exchanged between the browser UI and Aura peers.

use serde::{Deserialize, Serialize};

/// WebRTC / capture control messages.
///
/// The `type` tag is lowercased (`offer`, `answer`, `candidate`, ...). Candidate
/// fields accept both browser camelCase and snake_case for compatibility.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SignalingMessage {
    Offer {
        sdp: String,
    },
    Answer {
        sdp: String,
    },
    Candidate {
        candidate: String,
        #[serde(rename = "sdpMid", alias = "sdp_mid")]
        sdp_mid: Option<String>,
        #[serde(rename = "sdpMLineIndex", alias = "sdp_mline_index")]
        sdp_mline_index: Option<u32>,
    },
    Image {
        data: String,
    },
    TriggerImageCapture,
}
