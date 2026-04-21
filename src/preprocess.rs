use crate::image_io::GrayImage;
use anyhow::{bail, Result};
use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PolarityMode {
    Auto,
    Invert,
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preprocessed {
    pub contexts: Vec<u16>,
    pub symbols: Vec<u8>,
}

pub trait Preprocessor: Send + Sync {
    fn name(&self) -> &'static str;
    fn generate_contexts_and_symbols(
        &self,
        image: &GrayImage,
        invert: bool,
    ) -> Result<Preprocessed>;
}

pub fn normalize_polarity(pixels: &[u8], invert: bool) -> Vec<u8> {
    if invert {
        pixels.iter().map(|&px| 255u8.wrapping_sub(px)).collect()
    } else {
        pixels.to_vec()
    }
}

pub fn detect_polarity_auto(pixels: &[u8], width: usize, height: usize) -> bool {
    if width == 0 || height == 0 {
        return false;
    }

    let mut sum = 0u64;
    let mut count = 0u64;

    for &pixel in pixels.iter().take(width) {
        sum += pixel as u64;
        count += 1;
    }

    if height > 1 {
        let row = (height - 1) * width;
        for &pixel in pixels.iter().skip(row).take(width) {
            sum += pixel as u64;
            count += 1;
        }
    }

    if height > 2 {
        for y in 1..(height - 1) {
            sum += pixels[y * width] as u64;
            count += 1;
            if width > 1 {
                sum += pixels[y * width + width - 1] as u64;
                count += 1;
            }
        }
    }

    sum >= 128 * count
}

pub fn choose_polarity(image: &GrayImage, mode: PolarityMode) -> bool {
    match mode {
        PolarityMode::Auto => detect_polarity_auto(&image.pixels, image.width, image.height),
        PolarityMode::Invert => true,
        PolarityMode::None => false,
    }
}

pub fn ensure_preprocessed_lengths(image: &GrayImage, preprocessed: &Preprocessed) -> Result<()> {
    let symbols = image
        .width
        .checked_mul(image.height)
        .and_then(|px| px.checked_mul(8))
        .ok_or_else(|| anyhow::anyhow!("image symbol count overflow"))?;
    if preprocessed.contexts.len() != symbols || preprocessed.symbols.len() != symbols {
        bail!(
            "preprocessor produced invalid lengths: contexts {}, symbols {}, expected {}",
            preprocessed.contexts.len(),
            preprocessed.symbols.len(),
            symbols
        );
    }
    Ok(())
}
