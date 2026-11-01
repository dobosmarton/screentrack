//! Ambiguity rejection and cheapest-first, one-to-one matching.

use crate::candidates::Candidate;

/// Tracks and detections that have two near-equal candidates, so must not be matched.
#[derive(Debug, PartialEq)]
pub(crate) struct Ambiguity {
    pub(crate) tracks: Vec<bool>,
    pub(crate) detections: Vec<bool>,
}

/// The outcome of matching one frame.
#[derive(Debug, PartialEq)]
pub(crate) struct Matches {
    /// For each detection, the track it continues.
    pub(crate) by_detection: Vec<Option<usize>>,
    /// Whether each track was matched.
    pub(crate) used_tracks: Vec<bool>,
}

/// How far the walk over one identity's candidates has got.
#[derive(Clone, Copy)]
enum Progress {
    Unseen,
    Cheapest(f64),
    Settled,
}

impl Progress {
    /// Takes the next cost, returning whether it ties the identity's cheapest.
    fn advance(&mut self, cost: f64, margin: f64) -> bool {
        match *self {
            Self::Unseen => {
                *self = Self::Cheapest(cost);
                false
            }
            Self::Cheapest(best) => {
                *self = Self::Settled;
                cost - best < margin
            }
            Self::Settled => false,
        }
    }
}

/// Marks ambiguous identities in one pass over `candidates`, which must be sorted.
///
/// Only an identity's two cheapest candidates count, and sorting makes those the first
/// two it sees. Worse options must not count: in a crowded frame nearly every identity
/// has some pair of similar ones, which rejected almost every match on real footage.
pub(crate) fn find_ambiguous(
    candidates: &[Candidate],
    margin: f64,
    track_count: usize,
    detection_count: usize,
) -> Ambiguity {
    let mut by_track = vec![Progress::Unseen; track_count];
    let mut by_detection = vec![Progress::Unseen; detection_count];
    let mut ambiguity = Ambiguity {
        tracks: vec![false; track_count],
        detections: vec![false; detection_count],
    };
    for candidate in candidates {
        ambiguity.tracks[candidate.track] |=
            by_track[candidate.track].advance(candidate.cost, margin);
        ambiguity.detections[candidate.detection] |=
            by_detection[candidate.detection].advance(candidate.cost, margin);
    }
    ambiguity
}

/// Matches cheapest-first, skipping ambiguous and already used identities.
pub(crate) fn greedy_matches(candidates: &[Candidate], ambiguity: &Ambiguity) -> Matches {
    let mut matches = Matches {
        by_detection: vec![None; ambiguity.detections.len()],
        used_tracks: vec![false; ambiguity.tracks.len()],
    };
    for candidate in candidates {
        let Candidate {
            track, detection, ..
        } = *candidate;
        if !ambiguity.detections[detection]
            && !ambiguity.tracks[track]
            && matches.by_detection[detection].is_none()
            && !matches.used_tracks[track]
        {
            matches.by_detection[detection] = Some(track);
            matches.used_tracks[track] = true;
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const MARGIN: f64 = 0.15;

    fn sorted(raw: &[(f64, usize, usize)]) -> Vec<Candidate> {
        let mut candidates: Vec<Candidate> = raw
            .iter()
            .map(|&(cost, track, detection)| Candidate {
                cost,
                track,
                detection,
            })
            .collect();
        candidates.sort_unstable_by(Candidate::ordering);
        candidates
    }

    fn indices(flags: &[bool]) -> Vec<usize> {
        (0..flags.len()).filter(|&index| flags[index]).collect()
    }

    fn cheapest_two_are_close(costs: &[f64]) -> bool {
        let mut sorted = costs.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        matches!(sorted.as_slice(), [first, second, ..] if second - first < MARGIN)
    }

    /// The definition of ambiguity, written without the one-pass walk.
    fn exhaustive(candidates: &[Candidate], tracks: usize, detections: usize) -> Ambiguity {
        let flags = |count: usize, key: fn(&Candidate) -> usize| -> Vec<bool> {
            (0..count)
                .map(|identity| {
                    let costs: Vec<f64> = candidates
                        .iter()
                        .filter(|item| key(item) == identity)
                        .map(|item| item.cost)
                        .collect();
                    cheapest_two_are_close(&costs)
                })
                .collect()
        };
        Ambiguity {
            tracks: flags(tracks, |item| item.track),
            detections: flags(detections, |item| item.detection),
        }
    }

    #[test]
    fn close_costs_below_a_clear_winner_do_not_count() {
        let candidates = sorted(&[(0.0, 0, 0), (0.50, 0, 1), (0.52, 0, 2), (0.54, 0, 3)]);
        let ambiguity = find_ambiguous(&candidates, MARGIN, 1, 4);
        assert_eq!(indices(&ambiguity.tracks), [] as [usize; 0]);
        assert_eq!(ambiguity, exhaustive(&candidates, 1, 4));
    }

    #[test]
    fn close_leaders_are_ambiguous_however_many_follow() {
        let candidates = sorted(&[(0.0, 0, 0), (0.10, 0, 1), (0.90, 0, 2)]);
        let ambiguity = find_ambiguous(&candidates, MARGIN, 1, 3);
        assert_eq!(indices(&ambiguity.tracks), [0]);
        assert_eq!(ambiguity, exhaustive(&candidates, 1, 3));
    }

    #[test]
    fn equal_costs_are_ambiguous() {
        let candidates = sorted(&[(0.0, 0, 0), (0.0, 0, 1), (0.0, 1, 0)]);
        let ambiguity = find_ambiguous(&candidates, MARGIN, 2, 2);
        assert_eq!(indices(&ambiguity.tracks), [0]);
        assert_eq!(indices(&ambiguity.detections), [0]);
    }

    #[test]
    fn the_exact_margin_is_not_ambiguous() {
        let candidates = sorted(&[(0.0, 0, 0), (0.15, 0, 1), (0.15, 1, 0)]);
        let ambiguity = find_ambiguous(&candidates, MARGIN, 2, 2);
        assert_eq!(indices(&ambiguity.tracks), [] as [usize; 0]);
        assert_eq!(indices(&ambiguity.detections), [] as [usize; 0]);
    }

    #[test]
    fn matching_is_cheapest_first_and_one_to_one() {
        let candidates = sorted(&[(0.1, 0, 0), (0.5, 0, 1), (0.9, 1, 0), (1.3, 1, 1)]);
        let ambiguity = find_ambiguous(&candidates, MARGIN, 2, 2);
        let matches = greedy_matches(&candidates, &ambiguity);
        assert_eq!(matches.by_detection, [Some(0), Some(1)]);
        assert_eq!(matches.used_tracks, [true, true]);
    }

    #[test]
    fn ambiguous_identities_are_never_matched() {
        let candidates = sorted(&[(0.1, 0, 0), (0.2, 1, 0), (0.1, 2, 1), (0.2, 2, 2)]);
        let ambiguity = find_ambiguous(&candidates, MARGIN, 3, 3);
        let matches = greedy_matches(&candidates, &ambiguity);
        assert_eq!(matches.by_detection, [None, None, None]);
        assert_eq!(matches.used_tracks, [false, false, false]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn one_pass_finds_the_same_ambiguity_as_the_oracle(
            raw in prop::collection::vec((0_u8..=40, 0_usize..6, 0_usize..6), 0..48),
        ) {
            // A coarse grid of costs makes exact ties and exact margins common.
            let costs: Vec<(f64, usize, usize)> = raw
                .into_iter()
                .map(|(step, track, detection)| (f64::from(step) * 0.05, track, detection))
                .collect();
            let candidates = sorted(&costs);
            prop_assert_eq!(
                find_ambiguous(&candidates, MARGIN, 6, 6),
                exhaustive(&candidates, 6, 6)
            );
        }
    }
}
