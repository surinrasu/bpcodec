use crate::context::context_id_for_pixel;
use crate::image_io::GrayImage;
use crate::preprocess::{normalize_polarity, Preprocessed, Preprocessor};
use anyhow::Result;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuPreprocessor;

impl Preprocessor for CpuPreprocessor {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn generate_contexts_and_symbols(
        &self,
        image: &GrayImage,
        invert: bool,
    ) -> Result<Preprocessed> {
        Ok(generate_contexts_and_symbols_cpu(image, invert))
    }
}

pub fn generate_contexts_and_symbols_cpu(image: &GrayImage, invert: bool) -> Preprocessed {
    let pixels = normalize_polarity(&image.pixels, invert);
    let len = image.width * image.height * 8;
    let mut contexts = Vec::with_capacity(len);
    let mut symbols = Vec::with_capacity(len);

    for bit_index in (0..=7).rev() {
        for y in 0..image.height {
            for x in 0..image.width {
                let idx = y * image.width + x;
                contexts.push(context_id_for_pixel(
                    &pixels,
                    image.width,
                    image.height,
                    x,
                    y,
                    bit_index,
                ));
                symbols.push((pixels[idx] >> bit_index) & 1);
            }
        }
    }

    Preprocessed { contexts, symbols }
}
