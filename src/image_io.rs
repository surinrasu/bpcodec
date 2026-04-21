use anyhow::{anyhow, bail, Context, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, ImageReader, Luma};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("empty images are not supported");
        }
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| anyhow!("image dimensions overflow"))?;
        if pixels.len() != expected {
            bail!(
                "pixel buffer length mismatch: got {}, expected {}",
                pixels.len(),
                expected
            );
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn checked_u16_dimensions(&self) -> Result<(u16, u16)> {
        if self.width > u16::MAX as usize || self.height > u16::MAX as usize {
            bail!(
                "image dimensions {}x{} exceed u16 file format limits",
                self.width,
                self.height
            );
        }
        Ok((self.width as u16, self.height as u16))
    }

    pub fn raw_bytes(&self) -> usize {
        self.pixels.len()
    }
}

pub fn load_grayscale(path: impl AsRef<Path>) -> Result<GrayImage> {
    let path = path.as_ref();
    let reader = ImageReader::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to detect image format for {}", path.display()))?;
    let decoded = reader
        .decode()
        .with_context(|| format!("failed to decode image {}", path.display()))?;
    dynamic_to_grayscale(decoded, path)
}

pub fn save_grayscale_png(path: impl AsRef<Path>, image: &GrayImage) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }

    let width = u32::try_from(image.width).context("image width exceeds PNG limits")?;
    let height = u32::try_from(image.height).context("image height exceeds PNG limits")?;
    let buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, image.pixels.clone())
            .ok_or_else(|| anyhow!("failed to create grayscale output buffer"))?;
    buffer
        .save_with_format(path, ImageFormat::Png)
        .with_context(|| format!("failed to write PNG {}", path.display()))
}

pub fn list_image_paths(input_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let input_dir = input_dir.as_ref();
    let mut paths = Vec::new();
    collect_image_paths(input_dir, &mut paths)
        .with_context(|| format!("failed to scan {}", input_dir.display()))?;
    paths.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    Ok(paths)
}

fn collect_image_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_image_paths(&path, paths)?;
        } else if file_type.is_file() && is_supported_image_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

pub fn is_supported_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase()),
        Some(ext) if matches!(ext.as_str(), "png" | "pgm" | "pnm" | "ppm" | "pbm")
    )
}

fn dynamic_to_grayscale(image: DynamicImage, path: &Path) -> Result<GrayImage> {
    let (width, height) = image.dimensions();
    let width = usize::try_from(width).context("image width does not fit usize")?;
    let height = usize::try_from(height).context("image height does not fit usize")?;

    match image {
        DynamicImage::ImageLuma8(img) => GrayImage::new(width, height, img.into_raw()),
        DynamicImage::ImageLumaA8(img) => {
            let mut pixels = Vec::with_capacity(width * height);
            for px in img.into_raw().chunks_exact(2) {
                if px[1] != 255 {
                    bail!("{} has non-opaque alpha", path.display());
                }
                pixels.push(px[0]);
            }
            GrayImage::new(width, height, pixels)
        }
        DynamicImage::ImageRgb8(img) => {
            let mut pixels = Vec::with_capacity(width * height);
            for px in img.into_raw().chunks_exact(3) {
                if px[0] != px[1] || px[0] != px[2] {
                    bail!("{} contains non-grayscale RGB pixels", path.display());
                }
                pixels.push(px[0]);
            }
            GrayImage::new(width, height, pixels)
        }
        DynamicImage::ImageRgba8(img) => {
            let mut pixels = Vec::with_capacity(width * height);
            for px in img.into_raw().chunks_exact(4) {
                if px[3] != 255 {
                    bail!("{} has non-opaque alpha", path.display());
                }
                if px[0] != px[1] || px[0] != px[2] {
                    bail!("{} contains non-grayscale RGBA pixels", path.display());
                }
                pixels.push(px[0]);
            }
            GrayImage::new(width, height, pixels)
        }
        other => bail!(
            "{} is not an accepted 8-bit grayscale, grayscale-alpha, RGB, or RGBA image (decoded color: {:?})",
            path.display(),
            other.color()
        ),
    }
}
