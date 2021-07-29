use criterion::{black_box, criterion_group, criterion_main, Criterion};

use jpeg_encoder::{Encoder, ColorType, SamplingFactor};
use std::time::Duration;

fn create_bench_img() -> (Vec<u8>, u16, u16) {
    let width = 2000;
    let height = 1800;

    let mut data = Vec::with_capacity(width * height * 3);

    for y in 0..height {
        for x in 0..width {
            data.push((x % 256) as u8);
            data.push((x % 256) as u8);
            data.push(((x * y) % 256) as u8);
        }
    }

    (data, width as u16, height as u16)
}

fn encode_rgb_100(res: &mut Vec<u8>, data: &[u8], width: u16, height: u16) {
    let encoder = Encoder::new(res, 100);
    encoder.encode(data, width, height, ColorType::Rgb).unwrap();
}

fn encode_rgb_4x1(res: &mut Vec<u8>, data: &[u8], width: u16, height: u16) {
    let mut encoder = Encoder::new(res, 80);
    encoder.set_sampling_factor(SamplingFactor::F_4_1);
    encoder.encode(data, width, height, ColorType::Rgb).unwrap();
}

fn encode_rgb_progressive(res: &mut Vec<u8>, data: &[u8], width: u16, height: u16) {
    let mut encoder = Encoder::new(res, 80);
    encoder.set_progressive(true);
    encoder.encode(data, width, height, ColorType::Rgb).unwrap();
}

fn encode_rgb_optimized(res: &mut Vec<u8>, data: &[u8], width: u16, height: u16) {
    let mut encoder = Encoder::new(res, 100);
    encoder.set_optimized_huffman_tables(true);
    encoder.encode(data, width, height, ColorType::Rgb).unwrap();
}

fn encode_rgb_optimized_progressive(res: &mut Vec<u8>, data: &[u8], width: u16, height: u16) {
    let mut encoder = Encoder::new(res, 100);
    encoder.set_optimized_huffman_tables(true);
    encoder.set_progressive(true);
    encoder.encode(data, width, height, ColorType::Rgb).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut res = Vec::with_capacity(2 * 1024 * 1024);
    let (data, width, height) = create_bench_img();

    let mut group = c.benchmark_group("encode rgb");
    group.measurement_time(Duration::from_secs(25));

    group.bench_function("encode rgb 100", |b| b.iter(|| encode_rgb_100(
        black_box(&mut res),
        black_box(&data),
        black_box(width),
        black_box(height),
    )));

    group.bench_function("encode rgb 4x1", |b| b.iter(|| encode_rgb_4x1(
        black_box(&mut res),
        black_box(&data),
        black_box(width),
        black_box(height),
    )));

    group.bench_function("encode rgb progressive", |b| b.iter(|| encode_rgb_progressive(
        black_box(&mut res),
        black_box(&data),
        black_box(width),
        black_box(height),
    )));

    group.bench_function("encode rgb optimized", |b| b.iter(|| encode_rgb_optimized(
        black_box(&mut res),
        black_box(&data),
        black_box(width),
        black_box(height),
    )));

    group.bench_function("encode rgb optimized progressive", |b| b.iter(|| encode_rgb_optimized_progressive(
        black_box(&mut res),
        black_box(&data),
        black_box(width),
        black_box(height),
    )));

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);