use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
    Throughput,
};

use jpeg_encoder::ImageBuffer;
use std::time::Duration;

fn create_bench_img(width: usize, height: usize) -> Vec<u8> {
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

    data
}

fn bench_image(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, image: impl ImageBuffer) {
    let mut res = core::array::from_fn(|_| Vec::with_capacity(usize::from(image.width())));
    group.bench_function(name, |b| {
        b.iter(|| {
            for y in 0..image.height() {
                res[0].clear();
                res[1].clear();
                res[2].clear();
                image.fill_buffers(y, black_box(&mut res));
            }
            black_box(&res);
        });
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    for width in [1001, 1920] {
        let height = 500;
        let rgb = create_bench_img(width as usize, height as usize);
        let rgba: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        let bgr: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        let bgra: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0], 255])
            .collect();

        macro_rules! compare {
            ($format:literal, $data:ident, $scalar:ident, $avx:ident, $portable:ident) => {{
                let mut group = c.benchmark_group(format!("ycbcr/{}/{width}", $format));
                group.measurement_time(Duration::from_secs(3));
                group.warm_up_time(Duration::from_secs(1));
                group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                bench_image(
                    &mut group,
                    "scalar",
                    jpeg_encoder::$scalar(&$data, width, height),
                );
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                if std::is_x86_feature_detected!("avx2") {
                    bench_image(
                        &mut group,
                        "avx2",
                        jpeg_encoder::$avx(&$data, width, height),
                    );
                }
                #[cfg(feature = "use_wide")]
                bench_image(
                    &mut group,
                    "portable",
                    jpeg_encoder::$portable(&$data, width, height),
                );
                group.finish();
            }};
        }
        compare!("rgb", rgb, RgbImage, RgbImageAVX2, RgbImageSimd);
        compare!("rgba", rgba, RgbaImage, RgbaImageAVX2, RgbaImageSimd);
        compare!("bgr", bgr, BgrImage, BgrImageAVX2, BgrImageSimd);
        compare!("bgra", bgra, BgraImage, BgraImageAVX2, BgraImageSimd);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
