//! Label smoothing: a confidence-weighted, time-decayed vote over recent evidence.

use crate::LabelId;
use crate::config::Config;

/// One detector label remembered by a track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Evidence {
    pub(crate) label: LabelId,
    pub(crate) confidence: f64,
    pub(crate) captured_at: f64,
}

/// Votes over `evidence`, oldest first. `current` is this frame's raw label.
pub(crate) fn vote(
    config: &Config,
    evidence: &[Evidence],
    current: LabelId,
) -> (LabelId, Option<f64>) {
    let Some(newest) = evidence.last().filter(|_| evidence.len() >= 2) else {
        return (current, None);
    };
    let scores = decayed_scores(config, evidence, newest.captured_at);
    let total = scores.iter().fold(0.0, |sum, &(_, score)| sum + score);
    if total == 0.0 {
        return (current, Some(0.0));
    }

    let (winner, winning_score) = first_maximum(&scores);
    if winner != current {
        let runner_up = scores
            .iter()
            .filter(|&&(label, _)| label != winner)
            .fold(0.0_f64, |best, &(_, score)| best.max(score));
        let supporting = evidence.iter().filter(|item| item.label == winner).count();
        if supporting < 2 || (winning_score - runner_up) / total < config.min_label_margin {
            return (current, Some(score_of(&scores, current) / total));
        }
    }
    (winner, Some(winning_score / total))
}

/// Scores per label, in the order each label first appears in `evidence`.
fn decayed_scores(config: &Config, evidence: &[Evidence], now: f64) -> Vec<(LabelId, f64)> {
    let mut scores: Vec<(LabelId, f64)> = Vec::with_capacity(evidence.len());
    for item in evidence {
        let age = now - item.captured_at;
        let weight = item.confidence * 2.0_f64.powf(-age / config.label_half_life_s);
        match scores.iter_mut().find(|(label, _)| *label == item.label) {
            Some((_, score)) => *score += weight,
            None => scores.push((item.label, weight)),
        }
    }
    scores
}

/// Ties go to the label that appeared first, which is what Python's `max()` does.
fn first_maximum(scores: &[(LabelId, f64)]) -> (LabelId, f64) {
    let mut best = scores[0];
    for &entry in &scores[1..] {
        if entry.1 > best.1 {
            best = entry;
        }
    }
    best
}

fn score_of(scores: &[(LabelId, f64)], label: LabelId) -> f64 {
    scores
        .iter()
        .find(|(candidate, _)| *candidate == label)
        .map_or(0.0, |&(_, score)| score)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHEEP: LabelId = 1;
    const VILLAGER: LabelId = 2;
    const TREE: LabelId = 3;

    fn evidence(label: LabelId, confidence: f64, captured_at: f64) -> Evidence {
        Evidence {
            label,
            confidence,
            captured_at,
        }
    }

    fn aoe2_vote(items: &[Evidence], current: LabelId) -> (LabelId, Option<f64>) {
        vote(&Config::aoe2(), items, current)
    }

    #[test]
    fn a_single_observation_has_no_agreement() {
        let items = [evidence(SHEEP, 0.8, 1.0)];
        assert_eq!(aoe2_vote(&items, SHEEP), (SHEEP, None));
        assert_eq!(aoe2_vote(&[], SHEEP), (SHEEP, None));
    }

    #[test]
    fn unanimous_evidence_agrees_fully() {
        let items = [evidence(SHEEP, 0.8, 1.0), evidence(SHEEP, 0.8, 1.2)];
        assert_eq!(aoe2_vote(&items, SHEEP), (SHEEP, Some(1.0)));
    }

    #[test]
    fn a_single_flicker_keeps_the_raw_label() {
        let items = [
            evidence(SHEEP, 0.8, 1.0),
            evidence(SHEEP, 0.8, 1.2),
            evidence(VILLAGER, 0.8, 1.3),
        ];
        let (label, agreement) = aoe2_vote(&items, VILLAGER);
        assert_eq!(label, VILLAGER);
        assert!(agreement.is_some_and(|value| value < 0.5));
    }

    #[test]
    fn a_winner_with_one_entry_cannot_override_the_raw_label() {
        let items = [evidence(SHEEP, 0.99, 1.19), evidence(VILLAGER, 0.01, 1.2)];
        let (label, agreement) = aoe2_vote(&items, VILLAGER);
        assert_eq!(label, VILLAGER);
        assert!(agreement.is_some_and(|value| value < 0.5));
    }

    #[test]
    fn a_winner_needs_the_minimum_margin_to_override() {
        let items = [
            evidence(SHEEP, 0.5, 1.0),
            evidence(SHEEP, 0.5, 1.0),
            evidence(VILLAGER, 0.9, 1.0),
        ];
        // Sheep leads 1.0 to 0.9, a margin of about 0.05.
        assert_eq!(aoe2_vote(&items, VILLAGER).0, VILLAGER);
    }

    #[test]
    fn a_clear_winner_overrides_the_raw_label() {
        let items = [
            evidence(SHEEP, 0.8, 1.0),
            evidence(SHEEP, 0.8, 1.0),
            evidence(VILLAGER, 0.2, 1.0),
        ];
        let (label, agreement) = aoe2_vote(&items, VILLAGER);
        assert_eq!(label, SHEEP);
        assert_eq!(agreement, Some(1.6 / 1.8));
    }

    #[test]
    fn the_margin_is_measured_against_the_runner_up() {
        // Sheep's lead over the runner-up decides, not its lead over the raw label.
        let items = [
            evidence(SHEEP, 0.5, 1.0),
            evidence(SHEEP, 0.5, 1.0),
            evidence(TREE, 0.9, 1.0),
            evidence(VILLAGER, 0.1, 1.0),
        ];
        assert_eq!(aoe2_vote(&items, VILLAGER).0, VILLAGER);
    }

    #[test]
    fn older_evidence_decays_by_half_every_half_life() {
        let items = [evidence(SHEEP, 1.0, 1.0), evidence(VILLAGER, 1.0, 1.4)];
        let (_, agreement) = aoe2_vote(&items, VILLAGER);
        let agreement = agreement.expect("two entries produce an agreement");
        assert!((agreement - 2.0 / 3.0).abs() < 1e-12, "{agreement}");
    }

    #[test]
    fn all_zero_confidence_keeps_the_raw_label() {
        let items = [
            evidence(SHEEP, 0.0, 1.0),
            evidence(SHEEP, 0.0, 1.1),
            evidence(VILLAGER, 0.0, 1.2),
        ];
        assert_eq!(aoe2_vote(&items, VILLAGER), (VILLAGER, Some(0.0)));
    }

    #[test]
    fn ties_go_to_the_label_seen_first() {
        // Only a zero margin lets a tied winner through to the output.
        let config = Config {
            min_label_margin: 0.0,
            ..Config::aoe2()
        };
        let items = [
            evidence(SHEEP, 0.5, 1.0),
            evidence(VILLAGER, 0.5, 1.0),
            evidence(SHEEP, 0.5, 1.2),
            evidence(VILLAGER, 0.5, 1.2),
            evidence(TREE, 0.1, 1.2),
        ];
        assert_eq!(vote(&config, &items, TREE).0, SHEEP);
    }
}
