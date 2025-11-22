use criterion::{black_box, criterion_group, criterion_main, Criterion};

use jpeg_encoder::fdct;
use std::time::Duration;

const INPUT1: [i16; 64] = [
    -70, -71, -70, -68, -67, -67, -67, -67, -72, -73, -72, -70, -69, -69, -68, -69, -75, -76, -74,
    -73, -73, -72, -71, -70, -77, -78, -77, -75, -76, -75, -73, -71, -78, -77, -77, -76, -79, -77,
    -76, -75, -78, -78, -77, -77, -77, -77, -78, -77, -79, -79, -78, -78, -78, -78, -79, -78, -80,
    -79, -78, -78, -81, -80, -78, -76,
];

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("fdct");
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(10));

    group.bench_function("default fdct", |b| {
        b.iter(|| {
            let mut input = INPUT1.clone();
            fdct(black_box(&mut input));
            black_box(&input);
        })
    });

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    group.bench_function("fdct avx2", |b| {
        b.iter(|| {
            use jpeg_encoder::fdct_avx2;

            let mut input = INPUT1.clone();
            fdct_avx2(black_box(&mut input));
            black_box(&input);
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
