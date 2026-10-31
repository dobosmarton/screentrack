//! Appearance signatures: per-channel colour statistics of a box.

use crate::{BoundingBox, Signature};

const CHANNELS: usize = 3;
const MAX_CHANNEL_VALUE: f64 = 255.0;

/// Computes `[mean_r, mean_g, mean_b, std_r, std_g, std_b]` for a box, scaled to `[0, 1]`.
///
/// `pixels` holds rows of tightly packed RGB bytes, `row_stride` bytes apart. Coordinates
/// truncate toward zero and clamp to the image, and an empty crop returns all zeros.
///
/// # Panics
///
/// Panics if `pixels` is too short for `width`, `height` and `row_stride`.
#[must_use]
pub fn rgb_mean_std(
    pixels: &[u8],
    width: usize,
    height: usize,
    row_stride: usize,
    bbox: BoundingBox,
) -> Signature {
    let [x0, y0, x1, y1] = bbox;
    let (left, right) = (pixel_coordinate(x0, width), pixel_coordinate(x1, width));
    let (top, bottom) = (pixel_coordinate(y0, height), pixel_coordinate(y1, height));
    if left >= right || top >= bottom {
        return [0.0; 6];
    }

    let mut sums = [0_u64; CHANNELS];
    let mut squares = [0_u64; CHANNELS];
    for row in top..bottom {
        let start = row * row_stride;
        let line = &pixels[start + left * CHANNELS..start + right * CHANNELS];
        for pixel in line.as_chunks::<CHANNELS>().0 {
            for (channel, &value) in pixel.iter().enumerate() {
                let value = u64::from(value);
                sums[channel] += value;
                squares[channel] += value * value;
            }
        }
    }

    let count = u64::try_from((right - left) * (bottom - top)).unwrap_or(u64::MAX);
    let mut signature = [0.0; 6];
    for channel in 0..CHANNELS {
        signature[channel] = scaled_mean(sums[channel], count);
        signature[channel + CHANNELS] = scaled_std(sums[channel], squares[channel], count);
    }
    signature
}

/// Truncates toward zero like Python's `int()`, then clamps to `[0, limit]`.
fn pixel_coordinate(value: f64, limit: usize) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "truncation is the contract"
    )]
    let truncated = value as i64;
    usize::try_from(truncated).map_or(0, |coordinate| coordinate.min(limit))
}

/// Two divisions on purpose: this order reproduces `NumPy`'s `mean() / 255` exactly.
fn scaled_mean(sum: u64, count: u64) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "sums of u8 stay below 2^53")]
    let mean = sum as f64 / count as f64;
    mean / MAX_CHANNEL_VALUE
}

/// Population standard deviation from an exact integer numerator.
fn scaled_std(sum: u64, squares: u64, count: u64) -> f64 {
    let count = u128::from(count);
    let sum = u128::from(sum);
    let numerator = count * u128::from(squares) - sum * sum;
    #[expect(
        clippy::cast_precision_loss,
        reason = "rounding here is within tolerance"
    )]
    let variance = numerator as f64 / (count * count) as f64;
    variance.sqrt() / MAX_CHANNEL_VALUE
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY: Signature = [0.0; 6];

    fn solid(width: usize, height: usize, rgb: [u8; 3]) -> Vec<u8> {
        rgb.repeat(width * height)
    }

    #[test]
    fn solid_colour_has_its_mean_and_no_deviation() {
        let pixels = solid(4, 3, [255, 0, 51]);
        let signature = rgb_mean_std(&pixels, 4, 3, 12, [0.0, 0.0, 4.0, 3.0]);
        assert_eq!(signature, [1.0, 0.0, 0.2, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn population_deviation_of_two_values() {
        let pixels = [0, 0, 0, 255, 255, 255];
        let signature = rgb_mean_std(&pixels, 2, 1, 6, [0.0, 0.0, 2.0, 1.0]);
        assert_eq!(signature, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn row_stride_skips_padding_between_rows() {
        let padded = [10, 20, 30, 99, 99, 40, 50, 60, 99, 99];
        let tight = [10, 20, 30, 40, 50, 60];
        let bbox = [0.0, 0.0, 1.0, 2.0];
        assert_eq!(
            rgb_mean_std(&padded, 1, 2, 5, bbox),
            rgb_mean_std(&tight, 1, 2, 3, bbox)
        );
    }

    #[test]
    fn coordinates_truncate_toward_zero() {
        let pixels = solid(4, 4, [200, 100, 50]);
        assert_eq!(
            rgb_mean_std(&pixels, 4, 4, 12, [-0.9, -0.9, 1.9, 1.9]),
            rgb_mean_std(&pixels, 4, 4, 12, [0.0, 0.0, 1.0, 1.0])
        );
    }

    #[test]
    fn boxes_outside_or_too_thin_are_empty() {
        let pixels = solid(4, 4, [255, 255, 255]);
        for bbox in [
            [-3.0, 0.0, -1.0, 2.0],
            [0.0, -3.0, 2.0, -1.0],
            [5.0, 0.0, 7.0, 2.0],
            [0.0, 5.0, 2.0, 7.0],
            [1.2, 0.0, 1.8, 2.0],
            [0.0, 1.2, 2.0, 1.8],
        ] {
            assert_eq!(rgb_mean_std(&pixels, 4, 4, 12, bbox), EMPTY, "{bbox:?}");
        }
    }

    #[test]
    fn boxes_clamp_to_the_image() {
        let pixels = solid(4, 4, [255, 255, 255]);
        let signature = rgb_mean_std(&pixels, 4, 4, 12, [2.0, 2.0, 40.0, 40.0]);
        assert_eq!(signature, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }
}
