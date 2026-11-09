//! Which labels name units, as opposed to buildings and resources.

use std::collections::HashSet;

use crate::LabelId;

/// Labels whose objects move. Every other label is a static object.
#[derive(Clone, Debug, Default)]
pub(crate) struct UnitLabels(HashSet<LabelId>);

impl UnitLabels {
    pub(crate) fn new(labels: impl IntoIterator<Item = LabelId>) -> Self {
        Self(labels.into_iter().collect())
    }

    /// Whether `label` names a unit. A track without a label is not one.
    pub(crate) fn contains(&self, label: Option<LabelId>) -> bool {
        label.is_some_and(|label| self.0.contains(&label))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_listed_labels_are_units() {
        let units = UnitLabels::new([3, 5]);
        assert!(units.contains(Some(3)));
        assert!(!units.contains(Some(4)));
        assert!(!units.contains(None));
    }
}
