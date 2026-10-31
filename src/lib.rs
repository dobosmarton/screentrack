//! Stable identities for objects detected in screen captures.
//!
//! A detector returns independent boxes for each frame. [`Tracker`] links them across
//! frames, so each object keeps one ID and a smoothed label. Appearance signatures come
//! from [`rgb_mean_std`], which is separate so callers can run it off their main thread.
//!
//! The decision rules are defined in `docs/SPEC.md` and checked against golden fixtures
//! from the Python implementation this crate replaces.

mod candidates;
mod config;
mod label;
mod matching;
mod signature;
mod tracker;

#[cfg(feature = "python")]
mod python;

pub use candidates::Candidate;
pub use config::Config;
pub use signature::rgb_mean_std;
#[cfg(feature = "trace")]
pub use tracker::Trace;
pub use tracker::{FrameStats, Tracker};

/// An opaque label. Bindings intern strings, so the core only compares integers.
pub type LabelId = u32;

/// Per-channel RGB means followed by standard deviations, each scaled to `[0, 1]`.
pub type Signature = [f64; 6];

/// A box as `[x0, y0, x1, y1]`, with `x0 < x1` and `y0 < y1` guaranteed by the caller.
pub type BoundingBox = [f64; 4];

/// How the camera moved since the previous frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Motion {
    /// The view did not move.
    Stable,
    /// The content moved by this many full pixels.
    Pan {
        /// Horizontal content displacement.
        dx: f64,
        /// Vertical content displacement.
        dy: f64,
    },
    /// The view is discontinuous, so all track history is dropped.
    Reset,
}

/// One detector box and the evidence that goes with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    /// Box position in the capture.
    pub bbox: BoundingBox,
    /// Detector label.
    pub label: LabelId,
    /// Detector confidence.
    pub confidence: f64,
    /// Appearance signature of the box contents.
    pub signature: Signature,
}

/// The tracker's result for one detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tracked {
    /// Stable track ID. IDs are never reused.
    pub track_id: u64,
    /// Label after smoothing over recent evidence.
    pub label: LabelId,
    /// Label the detector reported for this frame.
    pub raw_label: LabelId,
    /// Share of the label vote, or `None` with fewer than two evidence entries.
    pub agreement: Option<f64>,
}
