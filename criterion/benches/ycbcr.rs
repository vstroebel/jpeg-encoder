use criterion::{black_box, criterion_group, criterion_main, Criterion};

use jpeg_encoder::{ImageBuffer, RgbImage};
use std::time::Duration;

fn create_bench_img() -> (Vec<u8>, u16, u16) {
    let width = 1001;
    let height = 500;

    let mut data = Vec::with_capacity(width * height * 3);

    for y in 0..height {
        for x in 0..width {
            if (x * y) % 13 == 0 {
                data.push(0);
                data.push(0);
                data.push(0);
            } else if (x * y) % 17 == 0 {
                data.push(255);
                data.push(255);
                data.push(255);
            } else if (x * y) % 19 == 0 {
                data.push(255);
                data.push(0);
                data.push(0);
            } else if (x * y) % 21 == 0 {
                data.push(0);
                data.push(0);
                data.push(255);
            } else if (x * y) % 23 == 0 {
                data.push(0);
                data.push(255);
                data.push(0);
            } else if (x * y) % 25 == 0 {
                data.push(96);
                data.push(255);
                data.push(96);
            } else if (x * y) % 27 == 0 {
                data.push(255);
                data.push(96);
                data.push(96);
            } else if (x * y) % 29 == 0 {
                data.push(96);
                data.push(96);
                data.push(255);
            } else {
                data.push((x % 256) as u8);
                data.push((x % 256) as u8);
                data.push(((x * y) % 256) as u8);
            }
        }
    }

    (data, width as u16, height as u16)
}

fn criterion_benchmark(c: &mut Criterion) {
    let (data, width, height) = create_bench_img();
    let res1 = Vec::with_capacity(usize::from(width));
    let res2 = Vec::with_capacity(usize::from(width));
    let res3 = Vec::with_capacity(usize::from(width));
    let res4 = Vec::with_capacity(usize::from(width));

    let mut res = [res1, res2, res3, res4];

    let mut group = c.benchmark_group("ycbcr");
    group.measurement_time(Duration::from_secs(45));
    group.warm_up_time(Duration::from_secs(10));

    group.bench_function("default ycbcr", |b| {
        let image_buffer = RgbImage(&data, width, height);

        b.iter(|| {
            for y in 0..height {
                res[0].clear();
                res[1].clear();
                res[2].clear();
                image_buffer.fill_buffers(y, black_box(&mut res));
            }
            black_box(&res);
        });
    });

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    group.bench_function("ycbcr avx2", |b| {
        use jpeg_encoder::RgbImageAVX2;
        let image_buffer = RgbImageAVX2(&data, width, height);

        b.iter(|| {
            for y in 0..height {
                res[0].clear();
                res[1].clear();
                res[2].clear();
                image_buffer.fill_buffers(y, black_box(&mut res));
            }
            black_box(&res);
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
