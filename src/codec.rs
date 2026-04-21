use crate::arith::{ArithmeticDecoder, ArithmeticEncoder};
use crate::context::{context_id_for_pixel, TOTAL_CONTEXTS};
use crate::image_io::GrayImage;
use crate::model::Model;
use crate::preprocess::{choose_polarity, ensure_preprocessed_lengths, PolarityMode, Preprocessor};
use anyhow::{bail, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedImage {
    pub width: usize,
    pub height: usize,
    pub polarity_inverted: bool,
    pub model_hash: [u8; 32],
    pub stream: Vec<u8>,
}

pub fn encode_image(
    model: &Model,
    model_hash: [u8; 32],
    image: &GrayImage,
    polarity_mode: PolarityMode,
    preprocessor: &dyn Preprocessor,
) -> Result<EncodedImage> {
    image.checked_u16_dimensions()?;
    let invert = choose_polarity(image, polarity_mode);
    let preprocessed = preprocessor.generate_contexts_and_symbols(image, invert)?;
    ensure_preprocessed_lengths(image, &preprocessed)?;

    let mut encoder = ArithmeticEncoder::new();
    for (&context, &symbol) in preprocessed.contexts.iter().zip(&preprocessed.symbols) {
        let context = context as usize;
        if context >= TOTAL_CONTEXTS {
            bail!("preprocessor produced out-of-range context {}", context);
        }
        encoder.encode_bit(symbol, model.p_zero_q12[context])?;
    }

    Ok(EncodedImage {
        width: image.width,
        height: image.height,
        polarity_inverted: invert,
        model_hash,
        stream: encoder.finish(),
    })
}

pub fn decode_image(model: &Model, encoded: &EncodedImage) -> Result<GrayImage> {
    let expected_hash = model.hash();
    if encoded.model_hash != expected_hash {
        bail!("model hash mismatch");
    }
    decode_stream(
        model,
        encoded.width,
        encoded.height,
        encoded.polarity_inverted,
        &encoded.stream,
    )
}

pub fn decode_stream(
    model: &Model,
    width: usize,
    height: usize,
    polarity_inverted: bool,
    stream: &[u8],
) -> Result<GrayImage> {
    if width == 0 || height == 0 {
        bail!("encoded image has empty dimensions");
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("decoded image dimensions overflow"))?;
    let mut normalized = vec![0u8; pixel_count];
    let mut decoder = ArithmeticDecoder::new(stream);

    for bit_index in (0..=7).rev() {
        for y in 0..height {
            for x in 0..width {
                let context =
                    context_id_for_pixel(&normalized, width, height, x, y, bit_index) as usize;
                let bit = decoder.decode_bit(model.p_zero_q12[context])?;
                if bit == 1 {
                    normalized[y * width + x] |= 1u8 << bit_index;
                }
            }
        }
    }

    let pixels = if polarity_inverted {
        normalized
            .into_iter()
            .map(|value| 255u8.wrapping_sub(value))
            .collect()
    } else {
        normalized
    };
    GrayImage::new(width, height, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::preprocess_cpu::CpuPreprocessor;

    #[test]
    fn synthetic_images_roundtrip_losslessly() {
        let model = Model::uniform(16);
        let hash = model.hash();
        let preprocessor = CpuPreprocessor;
        for image in synthetic_images() {
            let encoded =
                encode_image(&model, hash, &image, PolarityMode::Auto, &preprocessor).unwrap();
            let decoded = decode_image(&model, &encoded).unwrap();
            assert_eq!(decoded, image);
        }
    }

    fn synthetic_images() -> Vec<GrayImage> {
        let mut images = vec![
            GrayImage::new(1, 1, vec![255]).unwrap(),
            GrayImage::new(1, 1, vec![0]).unwrap(),
            GrayImage::new(400, 400, vec![255; 400 * 400]).unwrap(),
            GrayImage::new(400, 400, vec![0; 400 * 400]).unwrap(),
        ];

        let mut diagonal = vec![255u8; 400 * 400];
        for i in 0..400 {
            diagonal[i * 400 + i] = 0;
        }
        images.push(GrayImage::new(400, 400, diagonal).unwrap());

        let checkerboard: Vec<u8> = (0..400 * 400)
            .map(|idx| {
                let x = idx % 400;
                let y = idx / 400;
                if (x + y) % 2 == 0 {
                    0
                } else {
                    255
                }
            })
            .collect();
        images.push(GrayImage::new(400, 400, checkerboard).unwrap());

        let mut sparse = vec![255u8; 400 * 400];
        for y in (20..380).step_by(37) {
            for x in 10..390 {
                if (x + y) % 5 == 0 {
                    sparse[y * 400 + x] = 20 + ((x * y) % 80) as u8;
                }
            }
        }
        images.push(GrayImage::new(400, 400, sparse).unwrap());

        let mut state = 0xceda_1234_5678_9abcu64;
        let noise: Vec<u8> = (0..400 * 400)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        images.push(GrayImage::new(400, 400, noise).unwrap());

        images
    }
}
