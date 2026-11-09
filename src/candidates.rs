//! Candidate generation: plausible track-to-detection pairs and their costs.

use std::cmp::Ordering;
use std::ops::Range;

use crate::config::Config;
use crate::geometry::overlap;
use crate::tracker::Track;
use crate::units::UnitLabels;
use crate::{BoundingBox, Detection, LabelId, Signature};

/// A plausible pairing of an existing track with a detection in this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Lower is better: position, appearance distance and any label penalty.
    pub cost: f64,
    /// Index into the tracks that survived this frame's expiry.
    pub track: usize,
    /// Index into the frame's detections.
    pub detection: usize,
}

impl Candidate {
    /// Orders by cost, then track, then detection, so results are deterministic.
    pub(crate) fn ordering(&self, other: &Self) -> Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then(self.track.cmp(&other.track))
            .then(self.detection.cmp(&other.detection))
    }
}

/// How far a track may have moved by this frame.
struct MovementGate {
    center: (f64, f64),
    allowed: f64,
}

impl MovementGate {
    fn new(config: &Config, track: &Track, captured_at: f64) -> Self {
        let [x0, y0, x1, y1] = track.bbox;
        let size = (x1 - x0).max(y1 - y0).max(config.min_size_px);
        let elapsed = captured_at - track.last_seen_at;
        Self {
            center: center(&track.bbox),
            allowed: config.size_factor * size + config.speed_px_per_s * elapsed,
        }
    }

    /// The part of `by_x` within horizontal reach, inclusive at both ends.
    fn horizontal_range(&self, by_x: &[(f64, usize)]) -> Range<usize> {
        let (low, high) = (self.center.0 - self.allowed, self.center.0 + self.allowed);
        let start = by_x.partition_point(|&(x, _)| x < low);
        let end = by_x.partition_point(|&(x, _)| x <= high);
        start..end.max(start)
    }
}

/// The label a track last had, and whether it names a unit.
struct Previous {
    label: Option<LabelId>,
    unit: bool,
}

/// A detection's label, and whether it names a unit.
#[derive(Clone, Copy)]
struct Kind {
    label: LabelId,
    unit: bool,
}

/// Returns every feasible pair, sorted by [`Candidate::ordering`].
pub(crate) fn candidates(
    config: &Config,
    units: &UnitLabels,
    tracks: &[Track],
    detections: &[Detection],
    captured_at: f64,
) -> Vec<Candidate> {
    let kinds: Vec<Kind> = detections
        .iter()
        .map(|detection| Kind {
            label: detection.label,
            unit: units.contains(Some(detection.label)),
        })
        .collect();
    let mut by_x: Vec<(f64, usize)> = detections
        .iter()
        .enumerate()
        .map(|(index, detection)| (center(&detection.bbox).0, index))
        .collect();
    by_x.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut found = Vec::new();
    for (track_index, track) in tracks.iter().enumerate() {
        let gate = MovementGate::new(config, track, captured_at);
        let label = track.last_label();
        let previous = Previous {
            label,
            unit: units.contains(label),
        };
        for &(_, detection_index) in &by_x[gate.horizontal_range(&by_x)] {
            let detection = &detections[detection_index];
            let kind = kinds[detection_index];
            if let Some(cost) = matching_cost(config, track, &previous, detection, kind, &gate) {
                found.push(Candidate {
                    cost,
                    track: track_index,
                    detection: detection_index,
                });
            }
        }
    }
    found.sort_unstable_by(Candidate::ordering);
    found
}

/// Ranks a pair by overlap, looks and label, or refuses it.
///
/// Overlap separates neighbours far better than distance, because a tracked object
/// barely moves between frames. The label penalty stops a second label on one spot from
/// tying with the first, while a flickering label can still continue its track. Only
/// units may continue on a box they don't touch: otherwise a farm whose box was missed
/// for a frame passed its ID to a villager standing in it.
fn matching_cost(
    config: &Config,
    track: &Track,
    previous: &Previous,
    detection: &Detection,
    kind: Kind,
    gate: &MovementGate,
) -> Option<f64> {
    let (new_x, new_y) = center(&detection.bbox);
    let distance = py_hypot(new_x - gate.center.0, new_y - gate.center.1);
    if distance > gate.allowed {
        return None;
    }
    let appearance = signature_distance(&track.signature, &detection.signature);
    if appearance > config.max_appearance_distance {
        return None;
    }
    let shared = overlap(track.bbox, detection.bbox);
    let relabelled = previous.label.is_some_and(|label| label != kind.label);
    if relabelled && shared < config.relabel_overlap {
        return None;
    }
    let units = previous.unit && kind.unit;
    if previous.label.is_some() && !units && shared < config.static_overlap {
        return None;
    }
    let position = if shared > 0.0 {
        1.0 - shared
    } else {
        1.0 + distance / gate.allowed
    };
    let penalty = if relabelled {
        config.label_penalty
    } else {
        0.0
    };
    Some(position + appearance + penalty)
}

#[expect(clippy::manual_midpoint, reason = "mirrors the Python formula exactly")]
fn center(bbox: &BoundingBox) -> (f64, f64) {
    ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
}

/// A plain left-to-right sum, matching Python 3.11's `sum()` on floats.
fn signature_distance(a: &Signature, b: &Signature) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0, |sum, (x, y)| sum + (x - y) * (x - y))
        .sqrt()
}

const SPLIT: f64 = 134_217_729.0;

/// Python's `math.hypot` for two finite values.
///
/// This ports CPython 3.11's `vector_norm`. `f64::hypot` calls the platform's libm, which
/// disagrees with Python on about one input in ten and varies between platforms.
fn py_hypot(x: f64, y: f64) -> f64 {
    let values = [x.abs(), y.abs()];
    let max = values[0].max(values[1]);
    if max == 0.0 {
        return 0.0;
    }
    let exponent = frexp_exponent(max);
    if exponent < -1023 {
        return hypot_of_tiny(values, max);
    }

    let scale = power_of_two(-exponent);
    let mut sum = 1.0;
    let mut fractions = [0.0; 3];
    for value in values {
        let (high, low) = split(value * scale);
        add_tracked(&mut sum, &mut fractions[0], high * high);
        add_tracked(&mut sum, &mut fractions[1], 2.0 * high * low);
        fractions[2] += low * low;
    }
    let root = root_estimate(sum, fractions);

    let (high, low) = split(root);
    add_tracked(&mut sum, &mut fractions[0], -high * high);
    add_tracked(&mut sum, &mut fractions[1], -2.0 * high * low);
    add_tracked(&mut sum, &mut fractions[2], -low * low);
    let correction = sum - 1.0 + (fractions[0] + fractions[1] + fractions[2]);
    (root + correction / (2.0 * root)) / scale
}

/// A first approximation, which the correction step in `py_hypot` then refines.
fn root_estimate(sum: f64, fractions: [f64; 3]) -> f64 {
    (sum - 1.0 + (fractions[0] + fractions[1] + fractions[2])).sqrt()
}

/// CPython's path for inputs so small that the usual scale factor would overflow.
fn hypot_of_tiny(values: [f64; 2], max: f64) -> f64 {
    let mut sum = 1.0;
    let mut fraction = 0.0;
    for value in values {
        let ratio = value / max;
        add_tracked(&mut sum, &mut fraction, ratio * ratio);
    }
    max * (sum - 1.0 + fraction).sqrt()
}

/// Splits a value into high and low halves whose products are exact.
fn split(value: f64) -> (f64, f64) {
    let scaled = value * SPLIT;
    let high = scaled - (scaled - value);
    (high, value - high)
}

/// Adds `value` to `sum`, collecting the rounding error in `error`.
fn add_tracked(sum: &mut f64, error: &mut f64, value: f64) {
    let previous = *sum;
    *sum += value;
    *error += (previous - *sum) + value;
}

/// The exponent `frexp` returns for a positive finite value, with mantissa in `[0.5, 1)`.
fn frexp_exponent(value: f64) -> i64 {
    const MANTISSA_BITS: u32 = 52;
    let bits = value.to_bits();
    let biased = (bits >> MANTISSA_BITS).cast_signed();
    if biased == 0 {
        // Subnormal: the highest set bit gives the magnitude.
        i64::from(bits.ilog2()) - 1073
    } else {
        biased - 1022
    }
}

/// `2^exponent` for exponents from -1024 to 1023, exactly like `ldexp(1.0, exponent)`.
fn power_of_two(exponent: i64) -> f64 {
    if exponent >= -1022 {
        f64::from_bits((exponent + 1023).cast_unsigned() << 52)
    } else {
        f64::from_bits(1_u64 << (exponent + 1074).cast_unsigned())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::label::Evidence;

    const ZERO_SIGNATURE: Signature = [0.0; 6];
    const VILLAGER: LabelId = 0;
    const SHEEP: LabelId = 1;
    const HOUSE: LabelId = 2;
    const MILL: LabelId = 3;

    fn units() -> UnitLabels {
        UnitLabels::new([VILLAGER, SHEEP])
    }

    fn track(bbox: BoundingBox, signature: Signature, last_seen_at: f64) -> Track {
        Track {
            id: 0,
            bbox,
            signature,
            evidence: Vec::new(),
            last_seen_at,
        }
    }

    fn labelled(bbox: BoundingBox, label: LabelId) -> Track {
        Track {
            evidence: vec![Evidence {
                label,
                confidence: 0.8,
                captured_at: 1.0,
            }],
            ..track(bbox, ZERO_SIGNATURE, 1.0)
        }
    }

    fn detection(bbox: BoundingBox, signature: Signature) -> Detection {
        Detection {
            bbox,
            label: VILLAGER,
            confidence: 0.8,
            signature,
        }
    }

    fn of_kind(bbox: BoundingBox, label: LabelId) -> Detection {
        Detection {
            label,
            ..detection(bbox, ZERO_SIGNATURE)
        }
    }

    fn feasible_detections(tracks: &[Track], detections: &[Detection], at: f64) -> Vec<usize> {
        candidates(&Config::aoe2(), &units(), tracks, detections, at)
            .iter()
            .map(|item| item.detection)
            .collect()
    }

    fn only_cost(track: Track, detection: Detection) -> Option<f64> {
        let found = candidates(&Config::aoe2(), &units(), &[track], &[detection], 1.2);
        assert!(found.len() <= 1);
        found.first().map(|item| item.cost)
    }

    /// Scans every pair, skipping the horizontal pruning under test.
    fn exhaustive(
        config: &Config,
        units: &UnitLabels,
        tracks: &[Track],
        detections: &[Detection],
        at: f64,
    ) -> Vec<Candidate> {
        let mut found = Vec::new();
        for (track_index, track) in tracks.iter().enumerate() {
            let gate = MovementGate::new(config, track, at);
            let label = track.last_label();
            let previous = Previous {
                label,
                unit: units.contains(label),
            };
            for (detection_index, item) in detections.iter().enumerate() {
                let kind = Kind {
                    label: item.label,
                    unit: units.contains(Some(item.label)),
                };
                if let Some(cost) = matching_cost(config, track, &previous, item, kind, &gate) {
                    found.push(Candidate {
                        cost,
                        track: track_index,
                        detection: detection_index,
                    });
                }
            }
        }
        found.sort_unstable_by(Candidate::ordering);
        found
    }

    #[test]
    fn pruning_keeps_the_exact_movement_boundary() {
        // At one second, a 20-pixel box permits exactly 15 + 65 = 80 pixels.
        let tracks = [track([100.0, 50.0, 120.0, 70.0], ZERO_SIGNATURE, 1.0)];
        let detections = [
            detection([180.0, 50.0, 200.0, 70.0], ZERO_SIGNATURE),
            detection([180.01, 50.0, 200.01, 70.0], ZERO_SIGNATURE),
        ];
        assert_eq!(feasible_detections(&tracks, &detections, 2.0), [0]);
    }

    #[test]
    fn pruning_keeps_the_exact_boundary_on_the_left_too() {
        let tracks = [track([100.0, 50.0, 120.0, 70.0], ZERO_SIGNATURE, 1.0)];
        let detections = [
            detection([20.0, 50.0, 40.0, 70.0], ZERO_SIGNATURE),
            detection([19.99, 50.0, 39.99, 70.0], ZERO_SIGNATURE),
        ];
        assert_eq!(feasible_detections(&tracks, &detections, 2.0), [0]);
    }

    #[test]
    fn large_boxes_widen_the_gate() {
        // A 200-pixel box permits 150 + 65 = 215 pixels.
        let tracks = [track([100.0, 50.0, 300.0, 250.0], ZERO_SIGNATURE, 1.0)];
        let detections = [
            detection([315.0, 50.0, 515.0, 250.0], ZERO_SIGNATURE),
            detection([315.01, 50.0, 515.01, 250.0], ZERO_SIGNATURE),
        ];
        assert_eq!(feasible_detections(&tracks, &detections, 2.0), [0]);
    }

    #[test]
    fn the_box_size_floor_applies_to_tiny_tracks() {
        // A 2-pixel box still gets the 12-pixel floor: 9 + 65 = 74 pixels.
        let tracks = [track([0.0, 0.0, 2.0, 2.0], ZERO_SIGNATURE, 1.0)];
        let detections = [
            detection([74.0, 0.0, 76.0, 2.0], ZERO_SIGNATURE),
            detection([74.5, 0.0, 76.5, 2.0], ZERO_SIGNATURE),
        ];
        assert_eq!(feasible_detections(&tracks, &detections, 2.0), [0]);
    }

    #[test]
    fn vertical_distance_counts_even_inside_the_horizontal_range() {
        let tracks = [track([0.0, 0.0, 20.0, 20.0], ZERO_SIGNATURE, 1.0)];
        let detections = [detection([0.0, 500.0, 20.0, 520.0], ZERO_SIGNATURE)];
        assert_eq!(
            feasible_detections(&tracks, &detections, 1.2),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn appearance_gate_is_inclusive() {
        let reachable = [
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.56, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let tracks = [track([0.0, 0.0, 20.0, 20.0], ZERO_SIGNATURE, 1.0)];
        let detections = reachable.map(|signature| detection([0.0, 0.0, 20.0, 20.0], signature));
        assert_eq!(feasible_detections(&tracks, &detections, 1.2), [0]);
    }

    #[test]
    fn apart_boxes_cost_one_plus_normalized_distance_plus_appearance() {
        let tracks = [track([0.0, 0.0, 20.0, 20.0], ZERO_SIGNATURE, 1.0)];
        let signature = [0.3, 0.4, 0.0, 0.0, 0.0, 0.0];
        let detections = [detection([40.0, 0.0, 60.0, 20.0], signature)];
        let found = candidates(&Config::aoe2(), &units(), &tracks, &detections, 2.0);
        assert_eq!(found.len(), 1);
        assert!(
            (found[0].cost - (1.0 + 0.5 + 0.5)).abs() < 1e-12,
            "{}",
            found[0].cost
        );
    }

    #[test]
    fn overlapping_boxes_cost_one_minus_their_overlap() {
        let cost = only_cost(
            labelled([0.0, 0.0, 10.0, 10.0], HOUSE),
            of_kind([5.0, 0.0, 15.0, 10.0], HOUSE),
        );
        assert_eq!(cost, Some(1.0 - 1.0 / 3.0));
    }

    #[test]
    fn a_new_label_costs_the_penalty() {
        let cost = only_cost(
            labelled([0.0, 0.0, 10.0, 10.0], HOUSE),
            of_kind([0.0, 0.0, 10.0, 10.0], MILL),
        );
        assert_eq!(cost, Some(1.0));
    }

    #[test]
    fn a_new_label_needs_half_the_box() {
        let old = [0.0, 0.0, 30.0, 10.0];
        // Boxes inside the old one, overlapping by exactly 1/2 and just under it.
        let half = of_kind([15.0, 0.0, 30.0, 10.0], MILL);
        let under = of_kind([15.5, 0.0, 30.0, 10.0], MILL);
        assert!(only_cost(labelled(old, HOUSE), half).is_some());
        assert_eq!(only_cost(labelled(old, HOUSE), under), None);
    }

    #[test]
    fn a_static_track_needs_a_tenth_of_the_box() {
        let old = [0.0, 0.0, 11.0, 100.0];
        // Overlaps of exactly 1/10 and just under it, close enough for the movement gate.
        let tenth = of_kind([9.0, 0.0, 20.0, 100.0], HOUSE);
        let under = of_kind([9.1, 0.0, 20.1, 100.0], HOUSE);
        assert!(only_cost(labelled(old, HOUSE), tenth).is_some());
        assert_eq!(only_cost(labelled(old, HOUSE), under), None);
    }

    #[test]
    fn only_two_units_continue_without_touching() {
        let old = [0.0, 0.0, 30.0, 80.0];
        let beside = [32.0, 0.0, 62.0, 80.0];
        assert!(only_cost(labelled(old, VILLAGER), of_kind(beside, VILLAGER)).is_some());
        assert_eq!(
            only_cost(labelled(old, HOUSE), of_kind(beside, HOUSE)),
            None
        );
        // A label change needs overlap anyway, so a mixed pair can't continue apart.
        assert_eq!(
            only_cost(labelled(old, VILLAGER), of_kind(beside, SHEEP)),
            None
        );
    }

    #[test]
    fn a_track_without_a_label_continues_anywhere_in_reach() {
        let beside = of_kind([32.0, 0.0, 62.0, 80.0], HOUSE);
        assert!(only_cost(track([0.0, 0.0, 30.0, 80.0], ZERO_SIGNATURE, 1.0), beside).is_some());
    }

    #[test]
    fn candidate_work_is_spatially_bounded() {
        let config = Config::aoe2();
        // Mid-array, so both the start and the end of the range have to prune.
        let old = track([100_100.0, 50.0, 100_120.0, 70.0], ZERO_SIGNATURE, 1.0);
        let by_x: Vec<(f64, usize)> = (0..1000)
            .map(|index| {
                (
                    f64::from(index) * 200.0 + 110.0,
                    usize::try_from(index).unwrap(),
                )
            })
            .collect();
        let gate = MovementGate::new(&config, &old, 1.2);
        assert_eq!(gate.horizontal_range(&by_x).len(), 1);
    }

    #[test]
    fn hypot_matches_python_bit_for_bit() {
        let text = std::fs::read_to_string("fixtures/hypot.json").unwrap();
        let cases: Vec<[f64; 3]> = serde_json::from_str(&text).unwrap();
        assert_eq!(cases.len(), 4096);
        for [x, y, expected] in cases {
            let actual = py_hypot(x, y);
            assert_eq!(actual.to_bits(), expected.to_bits(), "hypot({x}, {y})");
        }
    }

    #[test]
    fn hypot_matches_python_over_a_million_inputs() {
        #[derive(serde::Deserialize)]
        struct Recorded {
            cases: u32,
            seed: u64,
            digest: String,
        }
        fn next_state(mut state: u64) -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^ (state << 17)
        }
        fn coordinate(state: u64) -> f64 {
            #[expect(clippy::cast_precision_loss, reason = "the value is 53 bits wide")]
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            unit * 1000.0 - 500.0
        }

        let text = std::fs::read_to_string("fixtures/hypot_digest.json").unwrap();
        let expected: Recorded = serde_json::from_str(&text).unwrap();
        let mut state = expected.seed;
        let mut digest = 0xCBF2_9CE4_8422_2325_u64;
        for _ in 0..expected.cases {
            state = next_state(state);
            let x = coordinate(state);
            state = next_state(state);
            let y = coordinate(state);
            digest = (digest ^ py_hypot(x, y).to_bits()).wrapping_mul(0x0000_0100_0000_01B3);
        }
        assert_eq!(format!("{digest:016x}"), expected.digest);
    }

    fn arb_signature() -> impl Strategy<Value = Signature> {
        prop::array::uniform6(0.0..0.2_f64)
    }

    fn arb_box() -> impl Strategy<Value = BoundingBox> {
        (0_i32..800, 0_i32..80, 1_i32..80, 1_i32..80).prop_map(|(x, y, width, height)| {
            let (x, y) = (f64::from(x) / 4.0, f64::from(y) / 4.0);
            [x, y, x + f64::from(width), y + f64::from(height)]
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn pruning_never_changes_the_candidate_set(
            old in prop::collection::vec((arb_box(), arb_signature(), 0_u8..5, 0_u32..4), 0..24),
            new in prop::collection::vec((arb_box(), arb_signature(), 0_u32..4), 0..24),
        ) {
            let config = Config::aoe2();
            let tracks: Vec<Track> = old
                .into_iter()
                .map(|(bbox, signature, age, label)| Track {
                    signature,
                    last_seen_at: 2.0 - f64::from(age) * 0.2,
                    ..labelled(bbox, label)
                })
                .collect();
            let detections: Vec<Detection> = new
                .into_iter()
                .map(|(bbox, signature, label)| Detection {
                    label,
                    ..detection(bbox, signature)
                })
                .collect();
            prop_assert_eq!(
                candidates(&config, &units(), &tracks, &detections, 2.0),
                exhaustive(&config, &units(), &tracks, &detections, 2.0)
            );
        }
    }
}
