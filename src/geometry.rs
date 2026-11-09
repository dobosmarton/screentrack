//! Box arithmetic shared by matching and pruning.

use crate::BoundingBox;

/// Intersection over union. Boxes that only touch have none.
pub(crate) fn overlap(a: BoundingBox, b: BoundingBox) -> f64 {
    let width = a[2].min(b[2]) - a[0].max(b[0]);
    let height = a[3].min(b[3]) - a[1].max(b[1]);
    if width <= 0.0 || height <= 0.0 {
        return 0.0;
    }
    let intersection = width * height;
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    intersection / (area_a + area_b - intersection)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_is_intersection_over_union() {
        assert_eq!(overlap([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]), 1.0);
        assert_eq!(
            overlap([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]),
            1.0 / 3.0
        );
        assert_eq!(
            overlap([0.0, 0.0, 10.0, 10.0], [0.0, 5.0, 10.0, 15.0]),
            1.0 / 3.0
        );
    }

    #[test]
    fn boxes_apart_on_one_axis_have_no_overlap() {
        assert_eq!(
            overlap([0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]),
            0.0
        );
        assert_eq!(
            overlap([0.0, 0.0, 10.0, 10.0], [0.0, 20.0, 10.0, 30.0]),
            0.0
        );
    }

    #[test]
    fn boxes_that_only_touch_have_no_overlap() {
        assert_eq!(
            overlap([0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]),
            0.0
        );
        assert_eq!(
            overlap([0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 10.0, 20.0]),
            0.0
        );
    }
}
