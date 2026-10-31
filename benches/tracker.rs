//! Benchmarks at steady state, where a crowded scene is at its slowest.
//!
//! Ambiguous tracks survive for a full history window, so the first frames understate
//! the cost. Each `observe` case warms up for several seconds of simulated frames first.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use screentrack::{BoundingBox, Config, Detection, Motion, Tracker, rgb_mean_std};

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const STRIDE: usize = WIDTH * 3;
const BOX_SIZE: f64 = 64.0;
const OBJECTS: u32 = 60;
const COLUMNS: u32 = 10;
const CROWDED_SPACING: f64 = 6.0;
const SEPARATED_SPACING: f64 = 100.0;
const WARMUP_FRAMES: i32 = 30;

/// A small deterministic generator, so runs compare the same scenes.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn jitter(&mut self) -> f64 {
        let step = u32::try_from(self.next() % 7).expect("below seven");
        f64::from(step) - 3.0
    }

    fn pixels(&mut self, length: usize) -> Vec<u8> {
        (0..length).map(|_| self.next().to_le_bytes()[0]).collect()
    }
}

fn scene(rng: &mut Rng, spacing: f64) -> Vec<BoundingBox> {
    (0..OBJECTS)
        .map(|index| {
            let x = 200.0 + f64::from(index % COLUMNS) * spacing + rng.jitter();
            let y = 200.0 + f64::from(index / COLUMNS) * spacing + rng.jitter();
            [x, y, x + BOX_SIZE, y + BOX_SIZE]
        })
        .collect()
}

fn detections(pixels: &[u8], boxes: &[BoundingBox]) -> Vec<Detection> {
    boxes
        .iter()
        .map(|&bbox| Detection {
            bbox,
            label: 0,
            confidence: 0.8,
            signature: rgb_mean_std(pixels, WIDTH, HEIGHT, STRIDE, bbox),
        })
        .collect()
}

fn time_of(frame: i32, rate_hz: u32) -> f64 {
    1.0 + f64::from(frame) / f64::from(rate_hz)
}

fn observe_case(criterion: &mut Criterion, name: &str, spacing: f64, rate_hz: u32) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let pixels = rng.pixels(STRIDE * HEIGHT);
    let mut warm = Tracker::new(Config::aoe2());
    for frame in 1..=WARMUP_FRAMES {
        let current = detections(&pixels, &scene(&mut rng, spacing));
        warm.observe(
            i64::from(frame),
            time_of(frame, rate_hz),
            Motion::Stable,
            &current,
        );
    }
    let next = detections(&pixels, &scene(&mut rng, spacing));
    let (capture_id, at) = (
        i64::from(WARMUP_FRAMES) + 1,
        time_of(WARMUP_FRAMES + 1, rate_hz),
    );
    criterion.bench_function(name, |bench| {
        bench.iter_batched_ref(
            || warm.clone(),
            |tracker| tracker.observe(capture_id, at, Motion::Stable, black_box(&next)),
            BatchSize::LargeInput,
        );
    });
}

fn observe(criterion: &mut Criterion) {
    observe_case(criterion, "observe/crowded_60_5hz", CROWDED_SPACING, 5);
    observe_case(criterion, "observe/crowded_60_10hz", CROWDED_SPACING, 10);
    observe_case(criterion, "observe/separated_60_5hz", SEPARATED_SPACING, 5);
}

fn signatures(criterion: &mut Criterion) {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let pixels = rng.pixels(STRIDE * HEIGHT);
    let boxes = scene(&mut rng, CROWDED_SPACING);
    criterion.bench_function("rgb_mean_std/60_boxes_64px", |bench| {
        bench.iter(|| {
            for &bbox in &boxes {
                black_box(rgb_mean_std(
                    black_box(&pixels),
                    WIDTH,
                    HEIGHT,
                    STRIDE,
                    bbox,
                ));
            }
        });
    });
}

criterion_group!(benches, signatures, observe);
criterion_main!(benches);
