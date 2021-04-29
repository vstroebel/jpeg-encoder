# JPEG encoder

[![docs.rs badge](https://docs.rs/jpeg-encoder/badge.svg)](https://docs.rs/jpeg-encoder/)
[![crates.io badge](https://img.shields.io/crates/v/jpeg-encoder.svg)](https://crates.io/crates/jpeg-encoder/)
[![Rust](https://github.com/vstroebel/jpeg-encoder/actions/workflows/rust.yml/badge.svg)](https://github.com/vstroebel/jpeg-encoder/actions/workflows/rust.yml)

JPEG encoder written in Rust

## Features

- Baseline and progressive compression
- Chroma subsampling
- Optimized huffman tables
- 1, 3 and 4 component colorspaces

## Example
```rust
use jpeg_encoder::{Encoder, ColorType};

// An array with 4 pixels in RGB format.
let data = [
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 255,
];

// Create new encoder that writes to a file with maximum quality (100)
let mut encoder = Encoder::new_file("some.jpeg", 100)?;

// Encode the data with dimension 2x2
encoder.encode(&data, 2, 2, ColorType::Rgb)?;
```

## Minimum Supported Version of Rust (MSRV)

This crate needs at lest 1.43 or higher.

## License

This project is licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted 
for inclusion in serde_urlencoded by you, as defined in the Apache-2.0 license, 
shall be dual licensed as above, without any additional terms or conditions.
