//! # Aura WebRTC Library
//!
//! A Rust library providing WebRTC signaling server and video streaming capabilities
//! for real-time communication applications.
//!
//! ## Features
//!
//! - **SignalingServer**: WebSocket-based signaling server for WebRTC peer connections
//! - **VideoStreamer**: Real-time video streaming with IVF file support
//! - **SignalingMessage**: Comprehensive message types for WebRTC signaling

use pyo3::prelude::*;

mod server;
mod signaling_types;
pub use signaling_types::SignalingMessage;

mod streamer;

#[cfg(test)]
mod tests;

use server::SignalingServer;
use streamer::VideoStreamer;

/// Python module initialization for the Aura WebRTC library.
///
/// This function registers the main classes with the Python interpreter:
/// - `SignalingServer`: For managing WebRTC signaling connections
/// - `VideoStreamer`: For streaming video content via WebRTC
#[pymodule]
fn aura(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SignalingServer>()?;
    m.add_class::<VideoStreamer>()?;
    Ok(())
}
