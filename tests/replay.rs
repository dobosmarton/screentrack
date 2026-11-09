//! Replays golden fixtures from the Python tracker and compares every decision.
//!
//! Decisions must match exactly. Scores may differ by [`TOLERANCE`], because Python's
//! `x ** 2` goes through libm `pow`. See `docs/SPEC.md` §6.1.

use std::collections::HashMap;
use std::path::Path;

use screentrack::{Candidate, Config, Detection, LabelId, Motion, Tracked, Tracker};
use serde::Deserialize;

const TOLERANCE: f64 = 1e-9;
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

#[derive(Deserialize)]
struct Manifest {
    mobile_labels: Vec<String>,
    tracker_sequences: Vec<String>,
}

#[derive(Deserialize)]
struct Sequence {
    frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Frame {
    capture_id: i64,
    captured_at: f64,
    motion: FixtureMotion,
    detections: Vec<FixtureDetection>,
    expected: Expected,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum FixtureMotion {
    Stable,
    Pan { dx: f64, dy: f64 },
    Reset,
}

#[derive(Deserialize)]
struct FixtureDetection {
    bbox: [f64; 4],
    label: String,
    confidence: f64,
    signature: [f64; 6],
}

#[derive(Deserialize)]
struct Expected {
    accepted: bool,
    candidate_count: usize,
    pair_digest: String,
    candidates: Vec<(f64, usize, usize)>,
    ambiguous_tracks: Vec<usize>,
    ambiguous_detections: Vec<usize>,
    ambiguous_count: usize,
    matches: Vec<(usize, usize)>,
    tracked: Vec<ExpectedTracked>,
    track_ids: Vec<u64>,
}

#[derive(Deserialize)]
struct ExpectedTracked {
    track_id: u64,
    label: String,
    raw_label: String,
    agreement: Option<f64>,
}

/// Interns labels as the Python bindings do, so strings stay out of the core.
#[derive(Default)]
struct Labels(HashMap<String, LabelId>);

impl Labels {
    fn id(&mut self, label: &str) -> LabelId {
        let next = LabelId::try_from(self.0.len()).expect("few labels");
        *self.0.entry(label.to_owned()).or_insert(next)
    }
}

impl From<FixtureMotion> for Motion {
    fn from(motion: FixtureMotion) -> Self {
        match motion {
            FixtureMotion::Stable => Self::Stable,
            FixtureMotion::Pan { dx, dy } => Self::Pan { dx, dy },
            FixtureMotion::Reset => Self::Reset,
        }
    }
}

/// The same fingerprint as the exporter: FNV-1a over `(track, detection)` words.
fn pair_digest(candidates: &[Candidate]) -> String {
    let mut pairs: Vec<u64> = candidates
        .iter()
        .map(|item| {
            let track = u64::try_from(item.track).expect("index fits in u64");
            let detection = u64::try_from(item.detection).expect("index fits in u64");
            track << 32 | detection
        })
        .collect();
    pairs.sort_unstable();
    let digest = pairs.iter().fold(FNV_OFFSET, |digest, pair| {
        (digest ^ pair).wrapping_mul(FNV_PRIME)
    });
    format!("{digest:016x}")
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{context}: {actual} vs {expected}"
    );
}

fn check_candidates(actual: &[Candidate], expected: &Expected, context: &str) {
    assert_eq!(actual.len(), expected.candidate_count, "{context}: count");
    assert_eq!(pair_digest(actual), expected.pair_digest, "{context}: set");
    let costs: HashMap<(usize, usize), f64> = actual
        .iter()
        .map(|item| ((item.track, item.detection), item.cost))
        .collect();
    for &(cost, track, detection) in &expected.candidates {
        let found = costs[&(track, detection)];
        assert_close(found, cost, context);
    }
}

fn check_output(
    actual: &[Tracked],
    expected: &[ExpectedTracked],
    labels: &mut Labels,
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}: output length");
    for (item, want) in actual.iter().zip(expected) {
        assert_eq!(item.track_id, want.track_id, "{context}: track ID");
        assert_eq!(item.label, labels.id(&want.label), "{context}: label");
        assert_eq!(item.raw_label, labels.id(&want.raw_label), "{context}: raw");
        match (item.agreement, want.agreement) {
            (None, None) => {}
            (Some(actual), Some(expected)) => assert_close(actual, expected, context),
            (actual, expected) => panic!("{context}: agreement {actual:?} vs {expected:?}"),
        }
    }
}

fn replay(name: &str, units: &[String]) -> usize {
    let path = Path::new("fixtures/tracker").join(format!("{name}.json"));
    let text = std::fs::read_to_string(path).expect("the fixture exists");
    let sequence: Sequence = serde_json::from_str(&text).expect("the fixture is valid");
    let mut labels = Labels::default();
    let unit_ids: Vec<LabelId> = units.iter().map(|label| labels.id(label)).collect();
    let mut tracker = Tracker::new(Config::aoe2(), unit_ids);
    for (index, frame) in sequence.frames.iter().enumerate() {
        let context = format!("{name} frame {index}");
        let detections: Vec<Detection> = frame
            .detections
            .iter()
            .map(|item| Detection {
                bbox: item.bbox,
                label: labels.id(&item.label),
                confidence: item.confidence,
                signature: item.signature,
            })
            .collect();
        let accepted = tracker.accepts(frame.capture_id, frame.captured_at);
        let expected = &frame.expected;
        assert_eq!(accepted, expected.accepted, "{context}: guard");
        let output = tracker.observe(
            frame.capture_id,
            frame.captured_at,
            frame.motion.into(),
            &detections,
        );
        assert_eq!(tracker.track_ids(), expected.track_ids, "{context}: tracks");
        let Some(output) = output else {
            assert!(!accepted, "{context}: an accepted frame returned nothing");
            continue;
        };
        let trace = tracker.trace();
        check_candidates(&trace.candidates, expected, &context);
        assert_eq!(
            trace.ambiguous_tracks, expected.ambiguous_tracks,
            "{context}: ambiguous tracks"
        );
        assert_eq!(
            trace.ambiguous_detections, expected.ambiguous_detections,
            "{context}: ambiguous detections"
        );
        assert_eq!(
            tracker.stats().ambiguous_count,
            expected.ambiguous_count,
            "{context}: ambiguous count"
        );
        assert_eq!(trace.matches, expected.matches, "{context}: matches");
        check_output(&output, &expected.tracked, &mut labels, &context);
    }
    sequence.frames.len()
}

#[test]
fn every_fixture_replays_identically() {
    let text = std::fs::read_to_string("fixtures/manifest.json").unwrap();
    let manifest: Manifest = serde_json::from_str(&text).unwrap();
    let frames: usize = manifest
        .tracker_sequences
        .iter()
        .map(|name| replay(name, &manifest.mobile_labels))
        .sum();
    assert!(frames >= 180, "only {frames} frames were replayed");
}
