//! Fixtures are only meaningful if JSON floats survive parsing with every bit intact.

#[derive(serde::Deserialize)]
struct Wrapper {
    value: f64,
}

fn parse(text: &str) -> f64 {
    let wrapped = format!("{{\"value\":{text}}}");
    serde_json::from_str::<Wrapper>(&wrapped)
        .expect("every finite float parses")
        .value
}

#[test]
fn shortest_float_text_parses_to_the_same_bits() {
    // Rust's `{:?}` and Python's `repr` both print the shortest text that round-trips.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..200_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f64::from_bits(state);
        if value.is_finite() {
            let text = format!("{value:?}");
            assert_eq!(parse(&text).to_bits(), value.to_bits(), "{text}");
        }
    }
}

#[test]
fn values_in_the_trackers_own_range_parse_exactly() {
    let mut state = 0xD1B5_4A32_D192_ED03_u64;
    for _ in 0..200_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let mantissa = state >> 12;
        #[expect(clippy::cast_precision_loss, reason = "the value is 52 bits wide")]
        let unit = mantissa as f64 / (1_u64 << 52) as f64;
        for value in [unit, unit * 2000.0, unit * 1e6] {
            let text = format!("{value:?}");
            assert_eq!(parse(&text).to_bits(), value.to_bits(), "{text}");
        }
    }
}
