use crate::context::{TOTAL_CONTEXTS, TOTAL_PROB};
use crate::image_io::{load_grayscale, GrayImage};
use crate::preprocess::{choose_polarity, ensure_preprocessed_lengths, PolarityMode, Preprocessor};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const MODEL_MAGIC: &[u8; 4] = b"BPM1";
const MODEL_VERSION: u16 = 1;
const MODEL_HEADER_LEN: usize = 4 + 2 + 2 + 4 + 4 + 32;
const MODEL_LEN: usize = MODEL_HEADER_LEN + TOTAL_CONTEXTS * 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub alpha: u32,
    pub p_zero_q12: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct ModelFile {
    pub model: Model,
    pub bytes: Vec<u8>,
    pub hash: [u8; 32],
}

impl Model {
    pub fn new(alpha: u32, p_zero_q12: Vec<u16>) -> Result<Self> {
        if p_zero_q12.len() != TOTAL_CONTEXTS {
            bail!(
                "model table length mismatch: got {}, expected {}",
                p_zero_q12.len(),
                TOTAL_CONTEXTS
            );
        }
        if let Some((idx, value)) = p_zero_q12
            .iter()
            .enumerate()
            .find(|(_, &value)| value == 0 || value >= TOTAL_PROB as u16)
        {
            bail!(
                "model probability at context {} is out of range: {}",
                idx,
                value
            );
        }
        Ok(Self { alpha, p_zero_q12 })
    }

    pub fn uniform(alpha: u32) -> Self {
        Self {
            alpha,
            p_zero_q12: vec![(TOTAL_PROB / 2) as u16; TOTAL_CONTEXTS],
        }
    }

    pub fn from_counts(zero_counts: &[u64], one_counts: &[u64], alpha: u32) -> Result<Self> {
        if alpha == 0 {
            bail!("alpha must be at least 1");
        }
        if zero_counts.len() != TOTAL_CONTEXTS || one_counts.len() != TOTAL_CONTEXTS {
            bail!("count arrays must have {} entries", TOTAL_CONTEXTS);
        }

        let mut table = Vec::with_capacity(TOTAL_CONTEXTS);
        for idx in 0..TOTAL_CONTEXTS {
            let zero = zero_counts[idx] as u128;
            let one = one_counts[idx] as u128;
            let alpha = alpha as u128;
            let den = zero + one + 2 * alpha;
            let num = (zero + alpha) * TOTAL_PROB as u128;
            let rounded = (num + den / 2) / den;
            let clamped = rounded.clamp(1, (TOTAL_PROB - 1) as u128) as u16;
            table.push(clamped);
        }

        Self::new(alpha, table)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MODEL_LEN);
        bytes.extend_from_slice(MODEL_MAGIC);
        bytes.extend_from_slice(&MODEL_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(TOTAL_PROB as u16).to_le_bytes());
        bytes.extend_from_slice(&(TOTAL_CONTEXTS as u32).to_le_bytes());
        bytes.extend_from_slice(&self.alpha.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
        for &prob in &self.p_zero_q12 {
            bytes.extend_from_slice(&prob.to_le_bytes());
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != MODEL_LEN {
            bail!(
                "invalid model length: got {}, expected {}",
                bytes.len(),
                MODEL_LEN
            );
        }
        let mut offset = 0usize;
        require_bytes(bytes, &mut offset, 4, "model magic").and_then(|magic| {
            if magic != MODEL_MAGIC {
                bail!("invalid model magic");
            }
            Ok(())
        })?;
        let version = read_u16(bytes, &mut offset, "model version")?;
        if version != MODEL_VERSION {
            bail!("unsupported model version {}", version);
        }
        let total_prob = read_u16(bytes, &mut offset, "model total_prob")?;
        if total_prob != TOTAL_PROB as u16 {
            bail!("invalid model total_prob {}", total_prob);
        }
        let total_contexts = read_u32(bytes, &mut offset, "model total_contexts")?;
        if total_contexts != TOTAL_CONTEXTS as u32 {
            bail!("invalid model total_contexts {}", total_contexts);
        }
        let alpha = read_u32(bytes, &mut offset, "model alpha")?;
        let reserved = require_bytes(bytes, &mut offset, 32, "model reserved")?;
        if reserved.iter().any(|&byte| byte != 0) {
            bail!("model reserved bytes must be zero");
        }

        let mut table = Vec::with_capacity(TOTAL_CONTEXTS);
        for _ in 0..TOTAL_CONTEXTS {
            table.push(read_u16(bytes, &mut offset, "model probability table")?);
        }
        if offset != bytes.len() {
            bail!("trailing bytes in model");
        }
        Self::new(alpha, table)
    }

    pub fn hash(&self) -> [u8; 32] {
        hash_model_bytes(&self.to_bytes())
    }
}

pub fn hash_model_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

pub fn read_model_file(path: impl AsRef<Path>) -> Result<ModelFile> {
    let path = path.as_ref();
    let bytes =
        fs::read(path).with_context(|| format!("failed to read model {}", path.display()))?;
    let model = Model::from_bytes(&bytes)
        .with_context(|| format!("failed to parse model {}", path.display()))?;
    let hash = hash_model_bytes(&bytes);
    Ok(ModelFile { model, bytes, hash })
}

pub fn write_model_file(path: impl AsRef<Path>, model: &Model) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create model directory {}", parent.display())
            })?;
        }
    }
    fs::write(path, model.to_bytes())
        .with_context(|| format!("failed to write model {}", path.display()))
}

pub fn train_model_from_paths(
    paths: &[PathBuf],
    alpha: u32,
    polarity_mode: PolarityMode,
    preprocessor: &dyn Preprocessor,
) -> Result<Model> {
    if paths.is_empty() {
        bail!("no supported images found");
    }

    let mut zero_counts = vec![0u64; TOTAL_CONTEXTS];
    let mut one_counts = vec![0u64; TOTAL_CONTEXTS];

    for path in paths {
        let image = load_grayscale(path)?;
        accumulate_image_counts(
            &image,
            alpha,
            polarity_mode,
            preprocessor,
            &mut zero_counts,
            &mut one_counts,
        )
        .with_context(|| format!("failed to train from {}", path.display()))?;
    }

    Model::from_counts(&zero_counts, &one_counts, alpha)
}

pub fn train_model_from_images(
    images: &[GrayImage],
    alpha: u32,
    polarity_mode: PolarityMode,
    preprocessor: &dyn Preprocessor,
) -> Result<Model> {
    if images.is_empty() {
        bail!("no images provided");
    }
    let mut zero_counts = vec![0u64; TOTAL_CONTEXTS];
    let mut one_counts = vec![0u64; TOTAL_CONTEXTS];

    for image in images {
        accumulate_image_counts(
            image,
            alpha,
            polarity_mode,
            preprocessor,
            &mut zero_counts,
            &mut one_counts,
        )?;
    }

    Model::from_counts(&zero_counts, &one_counts, alpha)
}

fn accumulate_image_counts(
    image: &GrayImage,
    alpha: u32,
    polarity_mode: PolarityMode,
    preprocessor: &dyn Preprocessor,
    zero_counts: &mut [u64],
    one_counts: &mut [u64],
) -> Result<()> {
    if alpha == 0 {
        bail!("alpha must be at least 1");
    }
    let invert = choose_polarity(image, polarity_mode);
    let preprocessed = preprocessor.generate_contexts_and_symbols(image, invert)?;
    ensure_preprocessed_lengths(image, &preprocessed)?;

    for (&context, &symbol) in preprocessed.contexts.iter().zip(&preprocessed.symbols) {
        let context = context as usize;
        if context >= TOTAL_CONTEXTS {
            bail!("preprocessor produced out-of-range context {}", context);
        }
        match symbol {
            0 => zero_counts[context] += 1,
            1 => one_counts[context] += 1,
            other => bail!("preprocessor produced invalid binary symbol {}", other),
        }
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: &mut usize, label: &str) -> Result<u16> {
    let raw = require_bytes(bytes, offset, 2, label)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn read_u32(bytes: &[u8], offset: &mut usize, label: &str) -> Result<u32> {
    let raw = require_bytes(bytes, offset, 4, label)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn require_bytes<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    len: usize,
    label: &str,
) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("{} length overflow", label))?;
    if end > bytes.len() {
        bail!("unexpected end of file while reading {}", label);
    }
    let out = &bytes[*offset..end];
    *offset = end;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess_cpu::CpuPreprocessor;

    #[test]
    fn model_serialization_roundtrip_preserves_hash() {
        let images = vec![
            GrayImage::new(2, 2, vec![255, 255, 0, 255]).unwrap(),
            GrayImage::new(2, 2, vec![0, 64, 128, 255]).unwrap(),
        ];
        let model =
            train_model_from_images(&images, 16, PolarityMode::Auto, &CpuPreprocessor).unwrap();
        let bytes = model.to_bytes();
        let hash = hash_model_bytes(&bytes);
        let parsed = Model::from_bytes(&bytes).unwrap();

        assert_eq!(model, parsed);
        assert_eq!(hash, parsed.hash());
        assert_eq!(bytes, parsed.to_bytes());
    }

    #[test]
    fn counts_to_probabilities_are_smoothed_and_clamped() {
        let mut zeros = vec![0u64; TOTAL_CONTEXTS];
        let mut ones = vec![0u64; TOTAL_CONTEXTS];
        zeros[0] = u64::MAX;
        ones[1] = u64::MAX;
        let model = Model::from_counts(&zeros, &ones, 16).unwrap();

        assert_eq!(model.p_zero_q12[0], 4095);
        assert_eq!(model.p_zero_q12[1], 1);
        assert_eq!(model.p_zero_q12[2], 2048);
    }
}
