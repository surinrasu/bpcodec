use crate::codec::{decode_image, decode_stream, encode_image};
use crate::container::{
    archive_to_bytes, read_archive_file, read_single_file, write_single_file, ArchiveEntry,
};
use crate::image_io::{list_image_paths, load_grayscale, save_grayscale_png};
use crate::model::{read_model_file, train_model_from_paths, write_model_file};
use crate::preprocess::{PolarityMode, Preprocessor};
use crate::preprocess_cpu::CpuPreprocessor;
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "bpcodec",
    version,
    about = "Strict lossless bit-plane image codec"
)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train {
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 16)]
        alpha: u32,
        #[arg(long, value_enum, default_value_t = PolarityMode::Auto)]
        polarity: PolarityMode,
    },
    Encode {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value_t = PolarityMode::Auto)]
        polarity: PolarityMode,
        #[arg(long)]
        use_metal: bool,
    },
    Decode {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Verify {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        use_metal: bool,
    },
    Pack {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        use_metal: bool,
    },
    Unpack {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        csv: PathBuf,
        #[arg(long)]
        use_metal: bool,
    },
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Train {
            input_dir,
            model,
            alpha,
            polarity,
        } => train_command(&input_dir, &model, alpha, polarity),
        Command::Encode {
            model,
            input,
            output,
            polarity,
            use_metal,
        } => encode_command(&model, &input, &output, polarity, use_metal),
        Command::Decode {
            model,
            input,
            output,
        } => decode_command(&model, &input, &output),
        Command::Verify {
            model,
            input,
            use_metal,
        } => verify_command(&model, &input, use_metal),
        Command::Pack {
            model,
            input_dir,
            output,
            use_metal,
        } => pack_command(&model, &input_dir, &output, use_metal),
        Command::Unpack { input, output_dir } => unpack_command(&input, &output_dir),
        Command::Bench {
            model,
            input_dir,
            csv,
            use_metal,
        } => bench_command(&model, &input_dir, &csv, use_metal),
    }
}

fn train_command(
    input_dir: &Path,
    model_path: &Path,
    alpha: u32,
    polarity: PolarityMode,
) -> Result<()> {
    let paths = list_image_paths(input_dir)?;
    let preprocessor = CpuPreprocessor;
    let model = train_model_from_paths(&paths, alpha, polarity, &preprocessor)?;
    write_model_file(model_path, &model)?;
    println!(
        "trained model: images={}, contexts={}, alpha={}, hash={}",
        paths.len(),
        model.p_zero_q12.len(),
        alpha,
        hex_hash(&model.hash())
    );
    Ok(())
}

fn encode_command(
    model_path: &Path,
    input: &Path,
    output: &Path,
    polarity: PolarityMode,
    use_metal: bool,
) -> Result<()> {
    let model_file = read_model_file(model_path)?;
    let image = load_grayscale(input)?;
    let preprocessor = make_preprocessor(use_metal)?;
    let encoded = encode_image(
        &model_file.model,
        model_file.hash,
        &image,
        polarity,
        preprocessor.as_ref(),
    )?;
    write_single_file(output, &encoded)?;
    println!(
        "encoded: {}x{}, stream_bytes={}, polarity={}, preprocessor={}",
        encoded.width,
        encoded.height,
        encoded.stream.len(),
        polarity_label(encoded.polarity_inverted),
        preprocessor.name()
    );
    Ok(())
}

fn decode_command(model_path: &Path, input: &Path, output: &Path) -> Result<()> {
    let model_file = read_model_file(model_path)?;
    let encoded = read_single_file(input)?;
    if encoded.model_hash != model_file.hash {
        bail!("model hash mismatch");
    }
    let decoded = decode_image(&model_file.model, &encoded)?;
    save_grayscale_png(output, &decoded)?;
    println!(
        "decoded: {}x{} -> {}",
        decoded.width,
        decoded.height,
        output.display()
    );
    Ok(())
}

fn verify_command(model_path: &Path, input: &Path, use_metal: bool) -> Result<()> {
    let model_file = read_model_file(model_path)?;
    let image = load_grayscale(input)?;
    let preprocessor = make_preprocessor(use_metal)?;

    let encode_start = Instant::now();
    let encoded = encode_image(
        &model_file.model,
        model_file.hash,
        &image,
        PolarityMode::Auto,
        preprocessor.as_ref(),
    )?;
    let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

    let decode_start = Instant::now();
    let decoded = decode_image(&model_file.model, &encoded)?;
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let verified = decoded == image;
    let bpp = bpp(encoded.stream.len(), image.width * image.height);

    println!(
        "compressed_bytes={}, bpp={:.6}, encode_ms={:.3}, decode_ms={:.3}, polarity={}, preprocessor={}, {}",
        encoded.stream.len(),
        bpp,
        encode_ms,
        decode_ms,
        polarity_label(encoded.polarity_inverted),
        preprocessor.name(),
        if verified { "PASS" } else { "FAIL" }
    );

    if !verified {
        bail!("roundtrip verification failed");
    }
    Ok(())
}

fn pack_command(model_path: &Path, input_dir: &Path, output: &Path, use_metal: bool) -> Result<()> {
    let model_file = read_model_file(model_path)?;
    let paths = list_image_paths(input_dir)?;
    if paths.is_empty() {
        bail!("no supported images found");
    }
    let preprocessor = make_preprocessor(use_metal)?;
    let preprocessor_ref = preprocessor.as_ref();

    let packed: Vec<PackResult> = paths
        .par_iter()
        .map(|path| {
            let image = load_grayscale(path)?;
            let name = relative_archive_name(input_dir, path)?;
            let raw_bytes = image.raw_bytes();
            let pixel_count = image.width * image.height;
            let start = Instant::now();
            let encoded = encode_image(
                &model_file.model,
                model_file.hash,
                &image,
                PolarityMode::Auto,
                preprocessor_ref,
            )?;
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(PackResult {
                entry: ArchiveEntry {
                    name,
                    width: encoded.width,
                    height: encoded.height,
                    polarity_inverted: encoded.polarity_inverted,
                    stream: encoded.stream,
                },
                raw_bytes,
                pixel_count,
                encode_ms,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let entries: Vec<ArchiveEntry> = packed.iter().map(|item| item.entry.clone()).collect();
    let archive_bytes = archive_to_bytes(&model_file.bytes, &entries)?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }
    fs::write(output, &archive_bytes)
        .with_context(|| format!("failed to write archive {}", output.display()))?;

    let total_raw: usize = packed.iter().map(|item| item.raw_bytes).sum();
    let total_pixels: usize = packed.iter().map(|item| item.pixel_count).sum();
    let total_encode_ms: f64 = packed.iter().map(|item| item.encode_ms).sum();
    println!(
        "packed: raw_bytes={}, archive_bytes={}, bpp={:.6}, avg_encode_ms={:.3}, images={}, preprocessor={}",
        total_raw,
        archive_bytes.len(),
        bpp(archive_bytes.len(), total_pixels),
        total_encode_ms / packed.len() as f64,
        packed.len(),
        preprocessor.name()
    );
    Ok(())
}

fn unpack_command(input: &Path, output_dir: &Path) -> Result<()> {
    let archive = read_archive_file(input)?;
    for entry in &archive.images {
        let decoded = decode_stream(
            &archive.model,
            entry.width,
            entry.height,
            entry.polarity_inverted,
            &entry.stream,
        )
        .with_context(|| format!("failed to decode archive image {}", entry.name))?;
        let output_path = output_dir.join(Path::new(&entry.name));
        save_grayscale_png(&output_path, &decoded)?;
    }
    println!(
        "unpacked: images={}, output_dir={}",
        archive.images.len(),
        output_dir.display()
    );
    Ok(())
}

fn bench_command(
    model_path: &Path,
    input_dir: &Path,
    csv_path: &Path,
    use_metal: bool,
) -> Result<()> {
    let model_file = read_model_file(model_path)?;
    let paths = list_image_paths(input_dir)?;
    if paths.is_empty() {
        bail!("no supported images found");
    }
    let preprocessor = make_preprocessor(use_metal)?;
    let preprocessor_ref = preprocessor.as_ref();

    let records: Vec<BenchRecord> = paths
        .par_iter()
        .map(|path| {
            let image = load_grayscale(path)?;
            let encode_start = Instant::now();
            let encoded = encode_image(
                &model_file.model,
                model_file.hash,
                &image,
                PolarityMode::Auto,
                preprocessor_ref,
            )?;
            let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

            let decode_start = Instant::now();
            let decoded = decode_image(&model_file.model, &encoded)?;
            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            let verified = decoded == image;
            Ok(BenchRecord {
                path: path.to_string_lossy().into_owned(),
                width: image.width,
                height: image.height,
                raw_bytes: image.raw_bytes(),
                compressed_bytes: encoded.stream.len(),
                bpp: bpp(encoded.stream.len(), image.width * image.height),
                encode_ms,
                decode_ms,
                polarity: polarity_label(encoded.polarity_inverted).to_owned(),
                verified,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    write_bench_csv(csv_path, &records)?;

    let total_raw: usize = records.iter().map(|record| record.raw_bytes).sum();
    let total_compressed: usize = records.iter().map(|record| record.compressed_bytes).sum();
    let total_pixels: usize = records
        .iter()
        .map(|record| record.width * record.height)
        .sum();
    let avg_encode_ms: f64 =
        records.iter().map(|record| record.encode_ms).sum::<f64>() / records.len() as f64;
    let avg_decode_ms: f64 =
        records.iter().map(|record| record.decode_ms).sum::<f64>() / records.len() as f64;
    let verified = records.iter().all(|record| record.verified);

    println!(
        "bench: images={}, raw_bytes={}, compressed_bytes={}, bpp={:.6}, avg_encode_ms={:.3}, avg_decode_ms={:.3}, verified={}, preprocessor={}",
        records.len(),
        total_raw,
        total_compressed,
        bpp(total_compressed, total_pixels),
        avg_encode_ms,
        avg_decode_ms,
        verified,
        preprocessor.name()
    );

    if !verified {
        bail!("one or more benchmark roundtrips failed");
    }
    Ok(())
}

fn make_preprocessor(use_metal: bool) -> Result<Box<dyn Preprocessor>> {
    if use_metal {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            return Ok(Box::new(crate::preprocess_metal::MetalPreprocessor::new()?));
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            bail!("--use-metal requires macOS and the Cargo feature 'metal'");
        }
    }
    Ok(Box::new(CpuPreprocessor))
}

fn relative_archive_name(input_dir: &Path, path: &Path) -> Result<String> {
    let relative = path.strip_prefix(input_dir).with_context(|| {
        format!(
            "failed to make {} relative to {}",
            path.display(),
            input_dir.display()
        )
    })?;
    let name = relative
        .to_str()
        .with_context(|| format!("path is not valid UTF-8: {}", relative.display()))?
        .replace(std::path::MAIN_SEPARATOR, "/");
    if name.is_empty() {
        bail!("empty archive path for {}", path.display());
    }
    Ok(name)
}

fn write_bench_csv(path: &Path, records: &[BenchRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create CSV directory {}", parent.display()))?;
        }
    }
    let mut file = fs::File::create(path)
        .with_context(|| format!("failed to create CSV {}", path.display()))?;
    writeln!(
        file,
        "path,width,height,raw_bytes,compressed_bytes,bpp,encode_ms,decode_ms,polarity,verified"
    )?;
    for record in records {
        writeln!(
            file,
            "{},{},{},{},{},{:.6},{:.3},{:.3},{},{}",
            csv_escape(&record.path),
            record.width,
            record.height,
            record.raw_bytes,
            record.compressed_bytes,
            record.bpp,
            record.encode_ms,
            record.decode_ms,
            csv_escape(&record.polarity),
            record.verified
        )?;
    }
    Ok(())
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn bpp(compressed_bytes: usize, pixels: usize) -> f64 {
    if pixels == 0 {
        0.0
    } else {
        compressed_bytes as f64 * 8.0 / pixels as f64
    }
}

fn polarity_label(inverted: bool) -> &'static str {
    if inverted {
        "invert"
    } else {
        "none"
    }
}

fn hex_hash(hash: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

struct PackResult {
    entry: ArchiveEntry,
    raw_bytes: usize,
    pixel_count: usize,
    encode_ms: f64,
}

struct BenchRecord {
    path: String,
    width: usize,
    height: usize,
    raw_bytes: usize,
    compressed_bytes: usize,
    bpp: f64,
    encode_ms: f64,
    decode_ms: f64,
    polarity: String,
    verified: bool,
}
