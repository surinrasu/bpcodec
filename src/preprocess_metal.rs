#![cfg(all(target_os = "macos", feature = "metal"))]

use crate::image_io::GrayImage;
use crate::preprocess::{Preprocessed, Preprocessor};
use crate::preprocess_cpu::generate_contexts_and_symbols_cpu;
use anyhow::Result;

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalPreprocessor;

impl MetalPreprocessor {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
}

impl Preprocessor for MetalPreprocessor {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn generate_contexts_and_symbols(
        &self,
        image: &GrayImage,
        invert: bool,
    ) -> Result<Preprocessed> {
        // TODO(metal): replace this CPU fallback with a macOS Metal kernel that emits
        // byte-identical contexts and symbols. Keeping the fallback behind the feature
        // gate preserves the public backend boundary without affecting the CPU decoder.
        Ok(generate_contexts_and_symbols_cpu(image, invert))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode_image;
    use crate::image_io::GrayImage;
    use crate::model::Model;
    use crate::preprocess::PolarityMode;
    use crate::preprocess_cpu::CpuPreprocessor;

    #[test]
    fn metal_contexts_and_symbols_match_cpu() {
        let image =
            GrayImage::new(4, 3, vec![255, 240, 0, 10, 64, 128, 192, 32, 1, 2, 3, 4]).unwrap();
        let metal = MetalPreprocessor::new().unwrap();
        let cpu = CpuPreprocessor;

        for invert in [false, true] {
            let from_cpu = cpu.generate_contexts_and_symbols(&image, invert).unwrap();
            let from_metal = metal.generate_contexts_and_symbols(&image, invert).unwrap();
            assert_eq!(from_metal, from_cpu);
        }
    }

    #[test]
    fn metal_and_cpu_preprocessing_produce_identical_streams() {
        let image = GrayImage::new(3, 3, vec![255, 255, 0, 255, 64, 0, 250, 230, 12]).unwrap();
        let model = Model::uniform(16);
        let hash = model.hash();
        let metal = MetalPreprocessor::new().unwrap();
        let cpu = CpuPreprocessor;

        let cpu_encoded = encode_image(&model, hash, &image, PolarityMode::Auto, &cpu).unwrap();
        let metal_encoded = encode_image(&model, hash, &image, PolarityMode::Auto, &metal).unwrap();

        assert_eq!(metal_encoded.stream, cpu_encoded.stream);
        assert_eq!(
            metal_encoded.polarity_inverted,
            cpu_encoded.polarity_inverted
        );
    }
}
