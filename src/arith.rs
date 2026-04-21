use crate::context::TOTAL_PROB;
use anyhow::{bail, Result};

const CODE_BITS: u32 = 32;
const TOP: u64 = (1u64 << CODE_BITS) - 1;
const HALF: u64 = 1u64 << (CODE_BITS - 1);
const QUARTER: u64 = HALF >> 1;
const THREE_QUARTERS: u64 = QUARTER * 3;

#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    bits_filled: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write_bit(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        self.current |= (bit & 1) << (7 - self.bits_filled);
        self.bits_filled += 1;
        if self.bits_filled == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.bits_filled = 0;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_filled > 0 {
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    pub fn read_bit(&mut self) -> u8 {
        if self.byte_pos >= self.bytes.len() {
            return 0;
        }
        let bit = (self.bytes[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        bit
    }
}

#[derive(Clone, Debug)]
pub struct ArithmeticEncoder {
    low: u64,
    high: u64,
    pending_bits: u64,
    writer: BitWriter,
}

impl Default for ArithmeticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithmeticEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            high: TOP,
            pending_bits: 0,
            writer: BitWriter::new(),
        }
    }

    pub fn encode_bit(&mut self, bit: u8, p_zero_q12: u16) -> Result<()> {
        validate_probability(p_zero_q12)?;
        if bit > 1 {
            bail!("arithmetic encoder received non-binary symbol {}", bit);
        }

        let range = self.high - self.low + 1;
        let zero_width = (range * p_zero_q12 as u64) / TOTAL_PROB as u64;
        let split = self.low + zero_width - 1;

        if bit == 0 {
            self.high = split;
        } else {
            self.low = split + 1;
        }

        self.renormalize();
        Ok(())
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.pending_bits += 1;
        if self.low < QUARTER {
            self.output_bit_plus_pending(0);
        } else {
            self.output_bit_plus_pending(1);
        }
        self.writer.finish()
    }

    fn renormalize(&mut self) {
        loop {
            if self.high < HALF {
                self.output_bit_plus_pending(0);
            } else if self.low >= HALF {
                self.output_bit_plus_pending(1);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTERS {
                self.pending_bits += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }

            self.low = (self.low << 1) & TOP;
            self.high = ((self.high << 1) | 1) & TOP;
        }
    }

    fn output_bit_plus_pending(&mut self, bit: u8) {
        self.writer.write_bit(bit);
        while self.pending_bits > 0 {
            self.writer.write_bit(1 - bit);
            self.pending_bits -= 1;
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArithmeticDecoder<'a> {
    low: u64,
    high: u64,
    code: u64,
    reader: BitReader<'a>,
}

impl<'a> ArithmeticDecoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        let mut reader = BitReader::new(bytes);
        let mut code = 0u64;
        for _ in 0..CODE_BITS {
            code = (code << 1) | reader.read_bit() as u64;
        }
        Self {
            low: 0,
            high: TOP,
            code,
            reader,
        }
    }

    pub fn decode_bit(&mut self, p_zero_q12: u16) -> Result<u8> {
        validate_probability(p_zero_q12)?;

        let range = self.high - self.low + 1;
        let scaled = ((((self.code - self.low + 1) as u128) * TOTAL_PROB as u128 - 1)
            / range as u128) as u16;
        let zero_width = (range * p_zero_q12 as u64) / TOTAL_PROB as u64;

        let bit = if scaled < p_zero_q12 {
            self.high = self.low + zero_width - 1;
            0
        } else {
            self.low += zero_width;
            1
        };

        self.renormalize();
        Ok(bit)
    }

    fn renormalize(&mut self) {
        loop {
            if self.high < HALF {
                // No state adjustment needed.
            } else if self.low >= HALF {
                self.code -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTERS {
                self.code -= QUARTER;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }

            self.low = (self.low << 1) & TOP;
            self.high = ((self.high << 1) | 1) & TOP;
            self.code = ((self.code << 1) | self.reader.read_bit() as u64) & TOP;
        }
    }
}

fn validate_probability(p_zero_q12: u16) -> Result<()> {
    if p_zero_q12 == 0 || p_zero_q12 >= TOTAL_PROB as u16 {
        bail!("invalid Q12 zero probability {}", p_zero_q12);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_writer_reader_are_msb_first() {
        let mut writer = BitWriter::new();
        for bit in [1, 0, 1, 1, 0, 0, 1, 0, 1] {
            writer.write_bit(bit);
        }
        let bytes = writer.finish();
        assert_eq!(bytes, vec![0b1011_0010, 0b1000_0000]);

        let mut reader = BitReader::new(&bytes);
        let mut bits = Vec::new();
        for _ in 0..16 {
            bits.push(reader.read_bit());
        }
        assert_eq!(bits, vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(reader.read_bit(), 0);
    }

    #[test]
    fn arithmetic_roundtrip_fixed_probability() {
        let bits = pseudo_random_bits(10_000);
        let probs = vec![2048u16; bits.len()];
        assert_roundtrip(&bits, &probs);
    }

    #[test]
    fn arithmetic_roundtrip_varying_probabilities() {
        let bits = pseudo_random_bits(20_000);
        let probs: Vec<u16> = (0..bits.len())
            .map(|i| 1 + ((i * 131 + 17) % 4095) as u16)
            .collect();
        assert_roundtrip(&bits, &probs);
    }

    #[test]
    fn arithmetic_roundtrip_extreme_probabilities() {
        for &prob in &[1u16, 2, 4094, 4095] {
            let bits = pseudo_random_bits(4096);
            let probs = vec![prob; bits.len()];
            assert_roundtrip(&bits, &probs);
        }
    }

    #[test]
    fn arithmetic_roundtrip_all_zeros_and_all_ones() {
        let zeros = vec![0u8; 5000];
        let ones = vec![1u8; 5000];
        let probs = vec![3072u16; zeros.len()];
        assert_roundtrip(&zeros, &probs);
        assert_roundtrip(&ones, &probs);
    }

    fn assert_roundtrip(bits: &[u8], probs: &[u16]) {
        let mut encoder = ArithmeticEncoder::new();
        for (&bit, &prob) in bits.iter().zip(probs) {
            encoder.encode_bit(bit, prob).unwrap();
        }
        let encoded = encoder.finish();

        let mut decoder = ArithmeticDecoder::new(&encoded);
        let mut decoded = Vec::with_capacity(bits.len());
        for &prob in probs {
            decoded.push(decoder.decode_bit(prob).unwrap());
        }
        assert_eq!(decoded, bits);
    }

    fn pseudo_random_bits(len: usize) -> Vec<u8> {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut bits = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            bits.push((state & 1) as u8);
        }
        bits
    }
}
