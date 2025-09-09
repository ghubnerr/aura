//! # Signaling Message Types
//!
//! This module defines the message types used for WebRTC signaling communication
//! between clients and the signaling server.

use serde::{Deserialize, Serialize};

/// WebRTC signaling messages for peer-to-peer communication setup.
///
/// This enum represents all possible message types that can be exchanged
/// during WebRTC connection establishment and maintenance.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SignalingMessage {
    /// WebRTC offer message containing SDP (Session Description Protocol) data.
    ///
    /// Sent by the initiating peer to propose connection parameters.
    Offer {
        /// Session Description Protocol data describing the offer
        sdp: String,
    },

    /// WebRTC answer message containing SDP data.
    ///
    /// Sent by the receiving peer in response to an offer.
    Answer {
        /// Session Description Protocol data describing the answer
        sdp: String,
    },

    /// ICE candidate information for NAT traversal.
    ///
    /// Contains network connectivity information to help establish
    /// the optimal connection path between peers.
    Candidate {
        /// ICE candidate string
        candidate: String,
        /// SDP media identification
        sdp_mid: Option<String>,
        /// SDP media line index
        sdp_mline_index: Option<u32>,
    },

    /// Image data message for screen capture or image sharing.
    ///
    /// Contains base64-encoded image data that can be shared between clients.
    Image {
        /// Base64-encoded image data (typically with data URL prefix)
        data: String,
    },

    /// Trigger message to request image capture from a client.
    ///
    /// Sent by the server to request that a client capture and send
    /// an image of their current screen or application state.
    TriggerImageCapture,
}
