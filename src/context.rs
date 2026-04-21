pub const TOTAL_PROB: u32 = 4096;
pub const TOTAL_CONTEXTS: usize = 5120;

pub fn context_id_for_pixel(
    pixels: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    bit_index: u8,
) -> u16 {
    debug_assert!(bit_index <= 7);
    debug_assert!(x < width);
    debug_assert!(y < height);
    debug_assert_eq!(pixels.len(), width * height);

    let plane_idx = 7usize - bit_index as usize;
    let pixel = pixels[y * width + x];
    let prefix_class = prefix_class(pixel, bit_index);

    let left_bit = if x > 0 {
        bit_at(pixels[y * width + x - 1], bit_index)
    } else {
        0
    };
    let up_bit = if y > 0 {
        bit_at(pixels[(y - 1) * width + x], bit_index)
    } else {
        0
    };
    let up_left_bit = if x > 0 && y > 0 {
        bit_at(pixels[(y - 1) * width + x - 1], bit_index)
    } else {
        0
    };
    let up_right_bit = if x + 1 < width && y > 0 {
        bit_at(pixels[(y - 1) * width + x + 1], bit_index)
    } else {
        0
    };

    let neigh_sum = left_bit + up_bit + up_left_bit + up_right_bit;
    let border_class = (x == 0) as usize + 2 * (y == 0) as usize;

    let context_id =
        (((((plane_idx * 8 + prefix_class) * 5 + neigh_sum as usize) * 2 + left_bit as usize) * 2
            + up_bit as usize)
            * 4)
            + border_class;
    debug_assert!(context_id < TOTAL_CONTEXTS);
    context_id as u16
}

#[inline]
pub fn bit_at(pixel: u8, bit_index: u8) -> u8 {
    (pixel >> bit_index) & 1
}

#[inline]
pub fn prefix_class(pixel: u8, bit_index: u8) -> usize {
    if bit_index == 7 {
        0
    } else {
        let lower_mask = (1u16 << (bit_index + 1)) - 1;
        let approx = (pixel as u16) & (!lower_mask & 0x00ff);
        usize::min(7, (approx >> 5) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_io::GrayImage;
    use crate::preprocess_cpu::generate_contexts_and_symbols_cpu;

    #[test]
    fn context_id_range_for_all_byte_values_and_positions() {
        let pixels: Vec<u8> = (0..25).map(|i| (i * 11) as u8).collect();
        for bit_index in 0..=7 {
            for y in 0..5 {
                for x in 0..5 {
                    let id = context_id_for_pixel(&pixels, 5, 5, x, y, bit_index);
                    assert!((id as usize) < TOTAL_CONTEXTS);
                }
            }
        }
    }

    #[test]
    fn known_3x3_context_ids() {
        let pixels = vec![0, 128, 255, 32, 64, 96, 127, 160, 224];

        assert_eq!(context_id_for_pixel(&pixels, 3, 3, 0, 0, 7), 3);
        assert_eq!(context_id_for_pixel(&pixels, 3, 3, 1, 0, 7), 2);
        assert_eq!(context_id_for_pixel(&pixels, 3, 3, 2, 1, 7), 36);

        let id = context_id_for_pixel(&pixels, 3, 3, 1, 1, 5);
        let plane_idx = 2usize;
        let prefix_class = 2usize;
        let neigh_sum = 2usize;
        let left_bit = 1usize;
        let up_bit = 0usize;
        let border_class = 0usize;
        let expected =
            (((((plane_idx * 8 + prefix_class) * 5 + neigh_sum) * 2 + left_bit) * 2 + up_bit) * 4)
                + border_class;
        assert_eq!(id as usize, expected);
    }

    #[test]
    fn cpu_generation_order_matches_bitplane_row_major() {
        let image = GrayImage::new(2, 2, vec![0b1000_0000, 0, 0, 0b0000_0001]).unwrap();
        let generated = generate_contexts_and_symbols_cpu(&image, false);

        assert_eq!(generated.symbols.len(), 32);
        assert_eq!(&generated.symbols[0..4], &[1, 0, 0, 0]);
        assert_eq!(&generated.symbols[28..32], &[0, 0, 0, 1]);
        assert_eq!(
            generated.contexts[0],
            context_id_for_pixel(&image.pixels, 2, 2, 0, 0, 7)
        );
        assert_eq!(
            generated.contexts[31],
            context_id_for_pixel(&image.pixels, 2, 2, 1, 1, 0)
        );
    }
}
