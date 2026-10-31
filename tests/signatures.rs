//! Appearance signatures against the values `NumPy` produced in the Python tracker.

use screentrack::rgb_mean_std;
use serde::Deserialize;

const CHANNELS: usize = 3;
const STD_TOLERANCE: f64 = 1e-9;

#[derive(Deserialize)]
struct Case {
    pixels: String,
    width: usize,
    height: usize,
    boxes: Vec<[f64; 4]>,
    expected: Vec<[f64; 6]>,
}

#[test]
fn signatures_match_numpy() {
    let text = std::fs::read_to_string("fixtures/signatures/cases.json").unwrap();
    let cases: Vec<Case> = serde_json::from_str(&text).unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let pixels = std::fs::read(format!("fixtures/signatures/{}", case.pixels)).unwrap();
        let stride = case.width * CHANNELS;
        assert_eq!(pixels.len(), stride * case.height);
        assert_eq!(case.boxes.len(), case.expected.len());
        for (bbox, expected) in case.boxes.iter().zip(&case.expected) {
            let actual = rgb_mean_std(&pixels, case.width, case.height, stride, *bbox);
            let context = format!("{} {bbox:?}", case.pixels);
            for channel in 0..CHANNELS {
                // The mean reproduces NumPy exactly; the deviation only within tolerance.
                assert_eq!(
                    actual[channel].to_bits(),
                    expected[channel].to_bits(),
                    "mean {context}"
                );
                let (std, want) = (actual[channel + CHANNELS], expected[channel + CHANNELS]);
                assert!(
                    (std - want).abs() <= STD_TOLERANCE,
                    "std {context}: {std} vs {want}"
                );
            }
        }
    }
}
