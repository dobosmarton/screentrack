//! The tracker state machine, following `docs/SPEC.md` §3.4 step by step.

use std::time::Instant;

#[cfg(feature = "trace")]
use crate::candidates::Candidate;
use crate::candidates::candidates;
use crate::config::Config;
use crate::geometry::overlap;
use crate::label::{Evidence, vote};
use crate::matching::{Matches, find_ambiguous, greedy_matches, with_refused_first_choices};
use crate::units::UnitLabels;
use crate::{BoundingBox, Detection, LabelId, Motion, Signature, Tracked};

const NEVER_CAPTURED_ID: i64 = -1;
const NEVER_CAPTURED_AT: f64 = -1.0;

/// How long tracks live without observations, in seconds.
#[derive(Clone, Copy)]
struct Lifetimes {
    unit: f64,
    static_object: f64,
}

/// An object remembered between frames.
#[derive(Clone, Debug)]
pub(crate) struct Track {
    pub(crate) id: u64,
    pub(crate) bbox: BoundingBox,
    pub(crate) signature: Signature,
    pub(crate) evidence: Vec<Evidence>,
    pub(crate) last_seen_at: f64,
}

impl Track {
    /// The label of the newest evidence, or `None` before any.
    pub(crate) fn last_label(&self) -> Option<LabelId> {
        self.evidence.last().map(|item| item.label)
    }
}

/// Measurements from the last accepted frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameStats {
    /// Detections with two near-equal candidates.
    pub ambiguous_count: usize,
    /// Feasible track-to-detection pairs.
    pub candidate_count: usize,
    /// Time spent generating candidates.
    pub candidate_ms: f64,
    /// Time spent on ambiguity, matching and output.
    pub matching_ms: f64,
}

/// Intermediate decisions from the last accepted frame, for parity tests.
#[cfg(feature = "trace")]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    /// Every feasible pair, sorted.
    pub candidates: Vec<Candidate>,
    /// Indices of ambiguous tracks, ascending.
    pub ambiguous_tracks: Vec<usize>,
    /// Indices of ambiguous detections, ascending.
    pub ambiguous_detections: Vec<usize>,
    /// `(detection, track)` index pairs, ordered by detection.
    pub matches: Vec<(usize, usize)>,
}

/// Links detections across frames. Keep one per video stream.
#[derive(Clone, Debug)]
pub struct Tracker {
    config: Config,
    units: UnitLabels,
    tracks: Vec<Track>,
    next_id: u64,
    last_capture_id: i64,
    last_at: f64,
    stats: FrameStats,
    #[cfg(feature = "trace")]
    trace: Trace,
}

impl Tracker {
    /// Creates an empty tracker.
    ///
    /// `unit_labels` names the labels of objects that move. Every other label is a
    /// static object, such as a building or a resource: its track lives longer and only
    /// continues on a box it touches.
    #[must_use]
    pub fn new(config: Config, unit_labels: impl IntoIterator<Item = LabelId>) -> Self {
        Self {
            config,
            units: UnitLabels::new(unit_labels),
            tracks: Vec::new(),
            next_id: 0,
            last_capture_id: NEVER_CAPTURED_ID,
            last_at: NEVER_CAPTURED_AT,
            stats: FrameStats::default(),
            #[cfg(feature = "trace")]
            trace: Trace::default(),
        }
    }

    /// Whether a frame is newer than the last one by both ID and time.
    #[must_use]
    pub fn accepts(&self, capture_id: i64, captured_at: f64) -> bool {
        capture_id > self.last_capture_id && captured_at > self.last_at
    }

    /// Processes one frame, returning a result for each detection in order.
    ///
    /// Returns `None`, and changes nothing, if [`Tracker::accepts`] rejects the frame.
    pub fn observe(
        &mut self,
        capture_id: i64,
        captured_at: f64,
        motion: Motion,
        detections: &[Detection],
    ) -> Option<Vec<Tracked>> {
        if !self.accepts(capture_id, captured_at) {
            return None;
        }
        if motion == Motion::Reset {
            self.reset();
        }
        let lifetimes = self.lifetimes(captured_at);
        self.last_capture_id = capture_id;
        self.last_at = captured_at;
        self.shift_and_expire(motion, captured_at, lifetimes);

        let candidate_started = Instant::now();
        let found = candidates(
            &self.config,
            &self.units,
            &self.tracks,
            detections,
            captured_at,
        );
        self.stats.candidate_ms = elapsed_ms(candidate_started);
        self.stats.candidate_count = found.len();

        let matching_started = Instant::now();
        let ambiguity = find_ambiguous(
            &found,
            self.config.ambiguity_margin,
            self.tracks.len(),
            detections.len(),
        );
        self.stats.ambiguous_count = ambiguity.detections.iter().filter(|&&flag| flag).count();
        let matches = greedy_matches(&found, &with_refused_first_choices(&found, &ambiguity));
        #[cfg(feature = "trace")]
        {
            self.trace = Trace::new(found, &ambiguity, &matches);
        }
        let tracked = self.emit(detections, matches, captured_at);
        self.stats.matching_ms = elapsed_ms(matching_started);
        Some(tracked)
    }

    /// Drops every track. The ID counter and the ordering guard are kept.
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Measurements from the last accepted frame.
    #[must_use]
    pub fn stats(&self) -> &FrameStats {
        &self.stats
    }

    /// Track IDs in list order, for parity tests.
    #[cfg(feature = "trace")]
    #[must_use]
    pub fn track_ids(&self) -> Vec<u64> {
        self.tracks.iter().map(|track| track.id).collect()
    }

    /// Intermediate decisions from the last accepted frame, for parity tests.
    #[cfg(feature = "trace")]
    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// How long unit and static tracks live without observations, as of this frame.
    ///
    /// Slow frames still need history: a fixed lifetime would forget everything at once.
    fn lifetimes(&self, captured_at: f64) -> Lifetimes {
        let config = &self.config;
        let unit = config
            .history_s
            .max(config.gap_factor * (captured_at - self.last_at));
        Lifetimes {
            unit,
            static_object: unit.max(config.static_memory_s),
        }
    }

    fn shift_and_expire(&mut self, motion: Motion, captured_at: f64, lifetimes: Lifetimes) {
        let units = &self.units;
        self.tracks.retain(|track| {
            let lifetime = if units.contains(track.last_label()) {
                lifetimes.unit
            } else {
                lifetimes.static_object
            };
            captured_at - track.last_seen_at <= lifetime
        });
        if let Motion::Pan { dx, dy } = motion {
            for track in &mut self.tracks {
                let [x0, y0, x1, y1] = track.bbox;
                track.bbox = [x0 + dx, y0 + dy, x1 + dx, y1 + dy];
            }
        }
    }

    /// Builds the output and the next track list: updated tracks in detection order,
    /// followed by unmatched old tracks in their previous order, without stale duplicates.
    fn emit(
        &mut self,
        detections: &[Detection],
        matches: Matches,
        captured_at: f64,
    ) -> Vec<Tracked> {
        let Matches {
            by_detection,
            used_tracks,
        } = matches;
        let mut output = Vec::with_capacity(detections.len());
        let mut updated = Vec::with_capacity(detections.len() + self.tracks.len());
        for (detection, matched) in detections.iter().zip(by_detection) {
            let (track_id, history) = match matched {
                Some(index) => {
                    let track = &mut self.tracks[index];
                    (track.id, std::mem::take(&mut track.evidence))
                }
                None => (self.allocate_id(), Vec::new()),
            };
            let evidence = self.extend_evidence(history, detection, captured_at);
            let (label, agreement) = vote(&self.config, &evidence, detection.label);
            output.push(Tracked {
                track_id,
                label,
                raw_label: detection.label,
                agreement,
            });
            updated.push(Track {
                id: track_id,
                bbox: detection.bbox,
                signature: detection.signature,
                evidence,
                last_seen_at: captured_at,
            });
        }
        let previous = std::mem::take(&mut self.tracks);
        let survivors: Vec<Track> = previous
            .into_iter()
            .zip(used_tracks)
            .filter(|(track, used)| !used && !self.is_stale_duplicate(track, &updated))
            .map(|(track, _)| track)
            .collect();
        updated.extend(survivors);
        self.tracks = updated;
        output
    }

    /// An unmatched track on top of a fresh one would keep that object ambiguous forever.
    ///
    /// Only the same label counts. The detector often puts two labels on one spot, and a
    /// second label that skips a frame should get its ID back when it returns.
    fn is_stale_duplicate(&self, track: &Track, updated: &[Track]) -> bool {
        let label = track.last_label();
        updated.iter().any(|fresh| {
            overlap(track.bbox, fresh.bbox) > self.config.stale_overlap
                && fresh.last_label() == label
        })
    }

    fn allocate_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn extend_evidence(
        &self,
        mut evidence: Vec<Evidence>,
        detection: &Detection,
        captured_at: f64,
    ) -> Vec<Evidence> {
        evidence.push(Evidence {
            label: detection.label,
            confidence: detection.confidence,
            captured_at,
        });
        evidence.retain(|item| captured_at - item.captured_at <= self.config.history_s);
        let excess = evidence.len().saturating_sub(self.config.max_observations);
        evidence.drain(..excess);
        evidence
    }
}

#[cfg(feature = "trace")]
impl Trace {
    fn new(
        candidates: Vec<Candidate>,
        ambiguity: &crate::matching::Ambiguity,
        matches: &Matches,
    ) -> Self {
        let set_indices = |flags: &[bool]| -> Vec<usize> {
            flags
                .iter()
                .enumerate()
                .filter_map(|(index, &flag)| flag.then_some(index))
                .collect()
        };
        Self {
            candidates,
            ambiguous_tracks: set_indices(&ambiguity.tracks),
            ambiguous_detections: set_indices(&ambiguity.detections),
            matches: matches
                .by_detection
                .iter()
                .enumerate()
                .filter_map(|(detection, track)| track.map(|track| (detection, track)))
                .collect(),
        }
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use super::*;

    const VILLAGER: u32 = 1;
    const SHEEP: u32 = 2;
    const HOUSE: u32 = 3;
    const MILL: u32 = 4;
    const FARM: u32 = 5;
    const BOX: BoundingBox = [100.0, 100.0, 120.0, 140.0];

    fn detection(label: u32, bbox: BoundingBox) -> Detection {
        Detection {
            bbox,
            label,
            confidence: 0.8,
            signature: [0.0; 6],
        }
    }

    fn track(bbox: BoundingBox) -> Track {
        Track {
            id: 0,
            bbox,
            signature: [0.0; 6],
            evidence: Vec::new(),
            last_seen_at: 0.0,
        }
    }

    fn tracker() -> Tracker {
        Tracker::new(Config::aoe2(), [VILLAGER, SHEEP])
    }

    fn step(tracker: &mut Tracker, id: i64, at: f64, detections: &[Detection]) -> Vec<Tracked> {
        tracker
            .observe(id, at, Motion::Stable, detections)
            .expect("the frame is newer")
    }

    #[test]
    fn pan_preserves_identity() {
        let mut tracks = tracker();
        let first = step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        let moved = detection(VILLAGER, [108.0, 104.0, 128.0, 144.0]);
        let pan = Motion::Pan { dx: 8.0, dy: 4.0 };
        let second = tracks.observe(2, 1.2, pan, &[moved]).expect("newer frame");
        assert_eq!(second[0].track_id, first[0].track_id);
        assert_eq!(step(&mut tracks, 3, 1.4, &[]), Vec::new());
    }

    #[test]
    fn pan_shifts_history_rather_than_the_new_box() {
        let mut tracks = tracker();
        step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        let far = detection(VILLAGER, [400.0, 100.0, 420.0, 140.0]);
        let pan = Motion::Pan { dx: 300.0, dy: 0.0 };
        let second = tracks.observe(2, 1.2, pan, &[far]).expect("newer frame");
        assert_eq!(second[0].track_id, 0);
    }

    #[test]
    fn class_flicker_reports_low_agreement() {
        let mut tracks = tracker();
        step(&mut tracks, 1, 1.0, &[detection(SHEEP, BOX)]);
        step(&mut tracks, 2, 1.2, &[detection(SHEEP, BOX)]);
        let flicker = step(&mut tracks, 3, 1.3, &[detection(VILLAGER, BOX)]);
        assert_eq!(flicker[0].label, VILLAGER);
        assert_eq!(flicker[0].raw_label, VILLAGER);
        assert!(flicker[0].agreement.is_some_and(|value| value < 1.0));
    }

    /// Steps through empty frames every 0.2 s, from `from` until just before `to`.
    fn empty_frames(tracker: &mut Tracker, first_id: i64, from: f64, to: f64) -> i64 {
        let mut id = first_id;
        let mut at = from;
        while at < to - 1e-9 {
            step(tracker, id, at, &[]);
            id += 1;
            at += 0.2;
        }
        id
    }

    #[test]
    fn history_expires_and_ids_are_not_reused() {
        let mut tracks = tracker();
        let first = step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        let id = empty_frames(&mut tracks, 2, 1.2, 2.2);
        let second = step(&mut tracks, id, 2.2, &[detection(VILLAGER, BOX)]);
        assert_ne!(first[0].track_id, second[0].track_id);
        assert_eq!(second[0].agreement, None);
    }

    #[test]
    fn a_unit_survives_at_exactly_the_history_limit() {
        let mut tracks = tracker();
        let first = step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        for (id, at) in [(2, 1.25), (3, 1.5), (4, 1.75)] {
            step(&mut tracks, id, at, &[]);
        }
        let later = step(&mut tracks, 5, 2.0, &[detection(VILLAGER, BOX)]);
        assert_eq!(later[0].track_id, first[0].track_id);
    }

    #[test]
    fn a_slow_frame_stretches_lifetimes_instead_of_resetting() {
        let mut tracks = tracker();
        let first = step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        let second = step(&mut tracks, 2, 3.0, &[detection(VILLAGER, BOX)]);
        assert_eq!(second[0].track_id, first[0].track_id);
    }

    #[test]
    fn lifetimes_stretch_to_exactly_three_frame_gaps() {
        let mut tracks = tracker();
        step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        step(&mut tracks, 2, 1.5, &[]);
        let lifetimes = tracks.lifetimes(2.0);
        assert_eq!(lifetimes.unit, 1.5);
        assert_eq!(lifetimes.static_object, 5.0);
        assert_eq!(tracks.lifetimes(4.0).static_object, 7.5);
    }

    #[test]
    fn a_missed_building_outlives_a_missed_unit() {
        let mut tracks = tracker();
        let house = [300.0, 100.0, 360.0, 160.0];
        let both = [detection(HOUSE, house), detection(VILLAGER, BOX)];
        let first = step(&mut tracks, 1, 1.0, &both);
        let id = empty_frames(&mut tracks, 2, 1.2, 6.0);
        let later = step(&mut tracks, id, 6.0, &both);
        assert_eq!(later[0].track_id, first[0].track_id);
        assert_ne!(later[1].track_id, first[1].track_id);
    }

    #[test]
    fn a_missed_building_is_forgotten_after_its_memory() {
        let mut tracks = tracker();
        let house = [detection(HOUSE, [300.0, 100.0, 360.0, 160.0])];
        let first = step(&mut tracks, 1, 1.0, &house);
        let id = empty_frames(&mut tracks, 2, 1.2, 6.2);
        let later = step(&mut tracks, id, 6.2, &house);
        assert_ne!(later[0].track_id, first[0].track_id);
    }

    #[test]
    fn two_labels_on_one_spot_keep_two_ids() {
        let mut tracks = tracker();
        let house = [300.0, 100.0, 360.0, 160.0];
        let both = [detection(HOUSE, house), detection(MILL, house)];
        let first = step(&mut tracks, 1, 1.0, &both);
        step(&mut tracks, 2, 1.2, &[detection(HOUSE, house)]);
        let again = step(&mut tracks, 3, 1.4, &both);
        assert_eq!(again[1].track_id, first[1].track_id);
    }

    #[test]
    fn a_farm_refused_its_box_never_takes_the_next_farm() {
        // Match 4, still clip, 48-50 s. The farm's ID was refused its own box as
        // ambiguous against a duplicate, and once took the next farm's box instead.
        let mut tracks = tracker();
        let farm = |bbox| detection(FARM, bbox);
        let first = step(&mut tracks, 1, 48.1, &[farm([190.0, 131.0, 465.0, 307.0])]);
        step(
            &mut tracks,
            2,
            48.3,
            &[
                farm([119.0, 106.0, 385.0, 277.0]),
                farm([268.0, 170.0, 499.0, 314.0]),
            ],
        );
        let last = step(
            &mut tracks,
            3,
            49.1,
            &[
                farm([58.0, 71.0, 318.0, 242.0]),
                farm([218.0, 148.0, 484.0, 316.0]),
            ],
        );
        assert_ne!(last[0].track_id, first[0].track_id);
    }

    #[test]
    fn occlusion_keeps_a_short_lived_identity() {
        let mut tracks = tracker();
        let first = step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        assert_eq!(step(&mut tracks, 2, 1.2, &[]), Vec::new());
        let third = step(&mut tracks, 3, 1.4, &[detection(VILLAGER, BOX)]);
        assert_eq!(third[0].track_id, first[0].track_id);
    }

    #[test]
    fn ambiguous_crossing_starts_a_new_track() {
        let mut tracks = tracker();
        let first = step(
            &mut tracks,
            1,
            1.0,
            &[
                detection(VILLAGER, BOX),
                detection(VILLAGER, [130.0, 100.0, 150.0, 140.0]),
            ],
        );
        let between = detection(VILLAGER, [115.0, 100.0, 135.0, 140.0]);
        let crossing = step(&mut tracks, 2, 1.2, &[between]);
        assert!(
            first
                .iter()
                .all(|item| item.track_id != crossing[0].track_id)
        );
        assert_eq!(tracks.stats().ambiguous_count, 1);
    }

    #[test]
    fn ambiguous_tracks_stay_for_later_frames() {
        let mut tracks = tracker();
        step(
            &mut tracks,
            1,
            1.0,
            &[
                detection(VILLAGER, BOX),
                detection(VILLAGER, [130.0, 100.0, 150.0, 140.0]),
            ],
        );
        let between = detection(VILLAGER, [115.0, 100.0, 135.0, 140.0]);
        step(&mut tracks, 2, 1.2, &[between]);
        assert_eq!(
            tracks
                .tracks
                .iter()
                .map(|track| track.id)
                .collect::<Vec<_>>(),
            [2, 0, 1]
        );
    }

    #[test]
    fn a_duplicate_box_heals_instead_of_cascading() {
        let object = detection(1, [100.0, 50.0, 120.0, 80.0]);
        let twin = detection(1, [102.0, 50.0, 122.0, 80.0]);
        let mut tracker = tracker();
        step(&mut tracker, 1, 1.0, &[object]);
        step(&mut tracker, 2, 1.2, &[object]);
        step(&mut tracker, 3, 1.4, &[object, twin]);
        let settled: Vec<u64> = [(4, 1.6), (5, 1.8), (6, 2.0), (7, 2.2), (8, 2.4)]
            .into_iter()
            .map(|(id, at)| step(&mut tracker, id, at, &[object])[0].track_id)
            .collect();
        assert!(settled[1..].windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn only_the_same_label_makes_a_stale_duplicate() {
        let tracker = tracker();
        let labelled = |label| Track {
            evidence: vec![Evidence {
                label,
                confidence: 0.8,
                captured_at: 0.0,
            }],
            ..track(BOX)
        };
        assert!(tracker.is_stale_duplicate(&labelled(HOUSE), &[labelled(HOUSE)]));
        assert!(!tracker.is_stale_duplicate(&labelled(MILL), &[labelled(HOUSE)]));
    }

    #[test]
    fn a_track_exactly_at_the_stale_limit_is_kept() {
        let tracker = tracker();
        let fresh = [track([0.0, 0.0, 10.0, 10.0])];
        assert!(!tracker.is_stale_duplicate(&track([0.0, 0.0, 10.0, 5.0]), &fresh));
        assert!(tracker.is_stale_duplicate(&track([0.0, 0.0, 10.0, 9.0]), &fresh));
    }

    #[test]
    fn repeated_evidence_corrects_an_initial_error() {
        let mut tracks = tracker();
        step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        step(&mut tracks, 2, 1.2, &[detection(SHEEP, BOX)]);
        let corrected = step(&mut tracks, 3, 1.4, &[detection(SHEEP, BOX)]);
        assert_eq!(corrected[0].label, SHEEP);
        assert_eq!(corrected[0].raw_label, SHEEP);
    }

    #[test]
    fn evidence_keeps_only_the_newest_observations() {
        let mut tracks = tracker();
        for (id, at) in [(1, 1.0), (2, 1.1), (3, 1.2), (4, 1.3)] {
            step(&mut tracks, id, at, &[detection(VILLAGER, BOX)]);
        }
        assert_eq!(tracks.tracks[0].evidence.len(), 3);
        assert_eq!(tracks.tracks[0].evidence[0].captured_at, 1.1);
    }

    #[test]
    fn evidence_older_than_the_window_is_dropped() {
        let config = Config {
            max_observations: 10,
            ..Config::aoe2()
        };
        let mut tracks = Tracker::new(config, [VILLAGER]);
        for (id, at) in [(1, 1.0), (2, 1.5), (3, 2.0), (4, 2.25)] {
            step(&mut tracks, id, at, &[detection(VILLAGER, BOX)]);
        }
        let times: Vec<f64> = tracks.tracks[0]
            .evidence
            .iter()
            .map(|item| item.captured_at)
            .collect();
        assert_eq!(times, [1.5, 2.0, 2.25]);
    }

    #[test]
    fn camera_reset_clears_history_without_reusing_an_id() {
        let mut tracks = tracker();
        let first = step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        let second = tracks
            .observe(2, 1.2, Motion::Reset, &[detection(VILLAGER, BOX)])
            .expect("newer frame");
        assert_ne!(second[0].track_id, first[0].track_id);
        assert!(tracks.observe(1, 1.0, Motion::Stable, &[]).is_none());
    }

    #[test]
    fn the_guard_needs_both_a_newer_id_and_a_newer_time() {
        let mut tracks = tracker();
        step(&mut tracks, 5, 5.0, &[]);
        assert!(!tracks.accepts(5, 6.0));
        assert!(!tracks.accepts(6, 5.0));
        assert!(!tracks.accepts(4, 6.0));
        assert!(tracks.accepts(6, 6.0));
    }

    #[test]
    fn capture_zero_is_accepted_first() {
        assert!(tracker().accepts(0, 0.0));
    }

    #[test]
    fn a_rejected_frame_changes_nothing() {
        let mut tracks = tracker();
        step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        let stats = *tracks.stats();
        assert_matches!(tracks.observe(1, 0.5, Motion::Reset, &[]), None);
        assert_eq!(tracks.tracks.len(), 1);
        assert_eq!(*tracks.stats(), stats);
    }

    #[test]
    fn reset_keeps_the_id_counter_and_the_guard() {
        let mut tracks = tracker();
        step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        tracks.reset();
        assert!(tracks.tracks.is_empty());
        assert!(!tracks.accepts(1, 1.0));
        let next = step(&mut tracks, 2, 1.2, &[detection(VILLAGER, BOX)]);
        assert_eq!(next[0].track_id, 1);
    }

    #[test]
    fn updated_tracks_come_first_then_unmatched_ones_in_order() {
        let mut tracks = tracker();
        let far = [500.0, 500.0, 520.0, 540.0];
        step(
            &mut tracks,
            1,
            1.0,
            &[detection(VILLAGER, BOX), detection(VILLAGER, far)],
        );
        step(&mut tracks, 2, 1.2, &[detection(VILLAGER, far)]);
        assert_eq!(
            tracks
                .tracks
                .iter()
                .map(|track| track.id)
                .collect::<Vec<_>>(),
            [1, 0]
        );
    }

    #[test]
    fn stats_count_the_frame() {
        let mut tracks = tracker();
        step(&mut tracks, 1, 1.0, &[detection(VILLAGER, BOX)]);
        step(&mut tracks, 2, 1.2, &[detection(VILLAGER, BOX)]);
        assert_eq!(tracks.stats().candidate_count, 1);
        assert_eq!(tracks.stats().ambiguous_count, 0);
        assert!(tracks.stats().candidate_ms >= 0.0);
        assert!(tracks.stats().matching_ms >= 0.0);
    }

    #[test]
    fn elapsed_time_is_reported_in_milliseconds() {
        let started = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(3));
        let elapsed = elapsed_ms(started);
        assert!((2.0..900.0).contains(&elapsed), "{elapsed}");
    }

    #[test]
    fn no_detections_is_a_valid_frame() {
        let mut tracks = tracker();
        assert_eq!(step(&mut tracks, 1, 1.0, &[]), Vec::new());
    }
}
