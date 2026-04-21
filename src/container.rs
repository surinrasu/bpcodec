use crate::codec::EncodedImage;
use crate::model::Model;
use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Component, Path};

const SINGLE_MAGIC: &[u8; 4] = b"BPC1";
const ARCHIVE_MAGIC: &[u8; 4] = b"BPA1";
const VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchiveEntry {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub polarity_inverted: bool,
    pub stream: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Archive {
    pub model: Model,
    pub model_bytes: Vec<u8>,
    pub images: Vec<ArchiveEntry>,
}

pub fn write_single_file(path: impl AsRef<Path>, encoded: &EncodedImage) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }
    fs::write(path, single_to_bytes(encoded)?)
        .with_context(|| format!("failed to write codec file {}", path.display()))
}

pub fn read_single_file(path: impl AsRef<Path>) -> Result<EncodedImage> {
    let path = path.as_ref();
    let bytes =
        fs::read(path).with_context(|| format!("failed to read codec file {}", path.display()))?;
    single_from_bytes(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

pub fn single_to_bytes(encoded: &EncodedImage) -> Result<Vec<u8>> {
    let width = u16::try_from(encoded.width).context("encoded width exceeds u16")?;
    let height = u16::try_from(encoded.height).context("encoded height exceeds u16")?;
    if width == 0 || height == 0 {
        bail!("encoded image has empty dimensions");
    }
    let stream_len = u64::try_from(encoded.stream.len()).context("stream length exceeds u64")?;
    let mut bytes = Vec::with_capacity(4 + 2 + 2 + 2 + 1 + 7 + 32 + 8 + encoded.stream.len());
    bytes.extend_from_slice(SINGLE_MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.push(encoded.polarity_inverted as u8);
    bytes.extend_from_slice(&[0u8; 7]);
    bytes.extend_from_slice(&encoded.model_hash);
    bytes.extend_from_slice(&stream_len.to_le_bytes());
    bytes.extend_from_slice(&encoded.stream);
    Ok(bytes)
}

pub fn single_from_bytes(bytes: &[u8]) -> Result<EncodedImage> {
    let mut parser = Parser::new(bytes);
    parser.expect_magic(SINGLE_MAGIC, "single-image magic")?;
    parser.expect_version()?;
    let width = parser.read_u16("width")? as usize;
    let height = parser.read_u16("height")? as usize;
    if width == 0 || height == 0 {
        bail!("encoded image has empty dimensions");
    }
    let flags = parser.read_u8("flags")?;
    if flags & !1 != 0 {
        bail!("single-image flags contain reserved bits: {flags:#04x}");
    }
    let reserved = parser.read_bytes(7, "reserved")?;
    if reserved.iter().any(|&byte| byte != 0) {
        bail!("single-image reserved bytes must be zero");
    }
    let mut model_hash = [0u8; 32];
    model_hash.copy_from_slice(parser.read_bytes(32, "model hash")?);
    let stream_len = parser.read_u64("stream length")?;
    let stream = parser.read_vec_len(stream_len, "stream")?;
    parser.finish()?;

    Ok(EncodedImage {
        width,
        height,
        polarity_inverted: flags & 1 != 0,
        model_hash,
        stream,
    })
}

pub fn write_archive_file(
    path: impl AsRef<Path>,
    model_bytes: &[u8],
    entries: &[ArchiveEntry],
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }
    fs::write(path, archive_to_bytes(model_bytes, entries)?)
        .with_context(|| format!("failed to write archive {}", path.display()))
}

pub fn read_archive_file(path: impl AsRef<Path>) -> Result<Archive> {
    let path = path.as_ref();
    let bytes =
        fs::read(path).with_context(|| format!("failed to read archive {}", path.display()))?;
    archive_from_bytes(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

pub fn archive_to_bytes(model_bytes: &[u8], entries: &[ArchiveEntry]) -> Result<Vec<u8>> {
    Model::from_bytes(model_bytes).context("embedded model bytes are not a valid BPM1 model")?;
    let model_len = u32::try_from(model_bytes.len()).context("model length exceeds u32")?;
    let image_count = u32::try_from(entries.len()).context("image count exceeds u32")?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(ARCHIVE_MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&model_len.to_le_bytes());
    bytes.extend_from_slice(model_bytes);
    bytes.extend_from_slice(&image_count.to_le_bytes());

    for entry in entries {
        validate_archive_name(&entry.name)?;
        let name_bytes = entry.name.as_bytes();
        let name_len =
            u16::try_from(name_bytes.len()).context("archive name length exceeds u16")?;
        let width = u16::try_from(entry.width).context("archive entry width exceeds u16")?;
        let height = u16::try_from(entry.height).context("archive entry height exceeds u16")?;
        if width == 0 || height == 0 {
            bail!("archive entry {} has empty dimensions", entry.name);
        }
        let stream_len =
            u64::try_from(entry.stream.len()).context("archive stream length exceeds u64")?;

        bytes.extend_from_slice(&name_len.to_le_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.push(entry.polarity_inverted as u8);
        bytes.extend_from_slice(&[0u8; 7]);
        bytes.extend_from_slice(&stream_len.to_le_bytes());
        bytes.extend_from_slice(&entry.stream);
    }

    Ok(bytes)
}

pub fn archive_from_bytes(bytes: &[u8]) -> Result<Archive> {
    let mut parser = Parser::new(bytes);
    parser.expect_magic(ARCHIVE_MAGIC, "archive magic")?;
    parser.expect_version()?;
    let model_len = parser.read_u32("model length")? as u64;
    let model_bytes = parser.read_vec_len(model_len, "embedded model")?;
    let model = Model::from_bytes(&model_bytes).context("invalid embedded model")?;
    let image_count = parser.read_u32("image count")? as usize;
    let mut images = Vec::with_capacity(image_count);

    for _ in 0..image_count {
        let name_len = parser.read_u16("name length")? as usize;
        let name_bytes = parser.read_bytes(name_len, "name")?;
        let name = std::str::from_utf8(name_bytes)
            .context("archive image name is not valid UTF-8")?
            .to_owned();
        validate_archive_name(&name)?;
        let width = parser.read_u16("width")? as usize;
        let height = parser.read_u16("height")? as usize;
        if width == 0 || height == 0 {
            bail!("archive entry {} has empty dimensions", name);
        }
        let flags = parser.read_u8("flags")?;
        if flags & !1 != 0 {
            bail!(
                "archive entry {} has reserved flag bits: {flags:#04x}",
                name
            );
        }
        let reserved = parser.read_bytes(7, "reserved")?;
        if reserved.iter().any(|&byte| byte != 0) {
            bail!("archive entry {} has nonzero reserved bytes", name);
        }
        let stream_len = parser.read_u64("stream length")?;
        let stream = parser.read_vec_len(stream_len, "stream")?;
        images.push(ArchiveEntry {
            name,
            width,
            height,
            polarity_inverted: flags & 1 != 0,
            stream,
        });
    }

    parser.finish()?;
    Ok(Archive {
        model,
        model_bytes,
        images,
    })
}

fn validate_archive_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("archive image name must not be empty");
    }
    if name.as_bytes().contains(&0) {
        bail!("archive image name contains NUL");
    }
    if name.contains('\\') {
        bail!("archive image name must use '/' separators: {}", name);
    }
    let path = Path::new(name);
    for component in path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("archive image name must be a safe relative path: {}", name)
            }
        }
    }
    Ok(())
}

struct Parser<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Parser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect_magic(&mut self, magic: &[u8; 4], label: &str) -> Result<()> {
        let found = self.read_bytes(4, label)?;
        if found != magic {
            bail!("invalid {}", label);
        }
        Ok(())
    }

    fn expect_version(&mut self) -> Result<()> {
        let version = self.read_u16("version")?;
        if version != VERSION {
            bail!("unsupported container version {}", version);
        }
        Ok(())
    }

    fn read_u8(&mut self, label: &str) -> Result<u8> {
        Ok(self.read_bytes(1, label)?[0])
    }

    fn read_u16(&mut self, label: &str) -> Result<u16> {
        let raw = self.read_bytes(2, label)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn read_u32(&mut self, label: &str) -> Result<u32> {
        let raw = self.read_bytes(4, label)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn read_u64(&mut self, label: &str) -> Result<u64> {
        let raw = self.read_bytes(8, label)?;
        Ok(u64::from_le_bytes([
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
        ]))
    }

    fn read_vec_len(&mut self, len: u64, label: &str) -> Result<Vec<u8>> {
        let len =
            usize::try_from(len).with_context(|| format!("{} length exceeds usize", label))?;
        Ok(self.read_bytes(len, label)?.to_vec())
    }

    fn read_bytes(&mut self, len: usize, label: &str) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("{} length overflow", label))?;
        if end > self.bytes.len() {
            bail!("unexpected end of file while reading {}", label);
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn finish(&self) -> Result<()> {
        if self.offset != self.bytes.len() {
            bail!("trailing bytes after container");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_image, decode_stream, encode_image};
    use crate::image_io::GrayImage;
    use crate::model::Model;
    use crate::preprocess::PolarityMode;
    use crate::preprocess_cpu::CpuPreprocessor;

    #[test]
    fn single_container_roundtrip() {
        let model = Model::uniform(16);
        let hash = model.hash();
        let image = GrayImage::new(3, 2, vec![255, 0, 127, 64, 32, 16]).unwrap();
        let encoded =
            encode_image(&model, hash, &image, PolarityMode::Auto, &CpuPreprocessor).unwrap();
        let bytes = single_to_bytes(&encoded).unwrap();
        let parsed = single_from_bytes(&bytes).unwrap();
        let decoded = decode_image(&model, &parsed).unwrap();
        assert_eq!(decoded, image);
    }

    #[test]
    fn archive_container_roundtrip() {
        let model = Model::uniform(16);
        let model_bytes = model.to_bytes();
        let hash = model.hash();
        let images = [
            (
                "a.png",
                GrayImage::new(2, 2, vec![255, 255, 0, 255]).unwrap(),
            ),
            (
                "nested/b.png",
                GrayImage::new(2, 2, vec![0, 64, 128, 255]).unwrap(),
            ),
        ];

        let mut entries = Vec::new();
        for (name, image) in &images {
            let encoded =
                encode_image(&model, hash, image, PolarityMode::Auto, &CpuPreprocessor).unwrap();
            entries.push(ArchiveEntry {
                name: (*name).to_owned(),
                width: encoded.width,
                height: encoded.height,
                polarity_inverted: encoded.polarity_inverted,
                stream: encoded.stream,
            });
        }

        let bytes = archive_to_bytes(&model_bytes, &entries).unwrap();
        let parsed = archive_from_bytes(&bytes).unwrap();
        assert_eq!(parsed.model, model);
        assert_eq!(parsed.images.len(), 2);

        for (entry, (_, original)) in parsed.images.iter().zip(images.iter()) {
            let decoded = decode_stream(
                &parsed.model,
                entry.width,
                entry.height,
                entry.polarity_inverted,
                &entry.stream,
            )
            .unwrap();
            assert_eq!(&decoded, original);
        }
    }
}
