//! Tuned values for the tracker.

/// Every tuned value used by [`Tracker`](crate::Tracker).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Track lifetime without observations, in seconds. Also the frame-gap reset
    /// threshold and the evidence window.
    pub history_s: f64,
    /// Allowed centre displacement per second of elapsed time.
    pub speed_px_per_s: f64,
    /// Allowed displacement as a fraction of the track's box size.
    pub size_factor: f64,
    /// Floor for box size in the movement gate.
    pub min_size_px: f64,
    /// Signature distance above which a pair is rejected.
    pub max_appearance_distance: f64,
    /// Cost difference below which two candidates count as a tie.
    pub ambiguity_margin: f64,
    /// Overlap above which an unmatched track is a stale duplicate of an updated one.
    pub stale_overlap: f64,
    /// Evidence entries kept per track.
    pub max_observations: usize,
    /// Decay half-life for label votes, in seconds.
    pub label_half_life_s: f64,
    /// Normalized margin required to switch a label.
    pub min_label_margin: f64,
}

impl Config {
    /// The values tuned for Age of Empires II at 1080p and above.
    #[must_use]
    pub const fn aoe2() -> Self {
        Self {
            history_s: 1.0,
            speed_px_per_s: 65.0,
            size_factor: 0.75,
            min_size_px: 12.0,
            max_appearance_distance: 0.55,
            ambiguity_margin: 0.15,
            stale_overlap: 0.5,
            max_observations: 3,
            label_half_life_s: 0.4,
            min_label_margin: 0.25,
        }
    }
}

impl Default for Config {
    /// The [`Config::aoe2`] preset, the only tuned configuration so far.
    fn default() -> Self {
        Self::aoe2()
    }
}
