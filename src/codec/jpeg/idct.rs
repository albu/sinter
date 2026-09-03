/// Fast 2D Integer Inverse Discrete Cosine Transform (IDCT) on 8x8 blocks
/// Based on standard Loeffler-Ligtenberg-Moschytz (LLM) integer algorithm

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

// Fixed-point constants scaled by 2^13 (8192)
const FIX_0_298631336: i32 = 2446;  // FIX(0.298631336)
const FIX_0_390180644: i32 = 3196;  // FIX(0.390180644)
const FIX_0_541196100: i32 = 4433;  // FIX(0.541196100)
const FIX_0_765366865: i32 = 6270;  // FIX(0.765366865)
const FIX_0_899976223: i32 = 7373;  // FIX(0.899976223)
const FIX_1_175875602: i32 = 9633;  // FIX(1.175875602)
const FIX_1_501321110: i32 = 12299; // FIX(1.501321110)
const FIX_1_847759065: i32 = 15137; // FIX(1.847759065)
const FIX_1_961570560: i32 = 16069; // FIX(1.961570560)
const FIX_2_053119869: i32 = 16819; // FIX(2.053119869)
const FIX_2_562915447: i32 = 20995; // FIX(2.562915447)
const FIX_3_072711026: i32 = 25172; // FIX(3.072711026)

#[inline(always)]
fn multiply(c: i32, x: i32) -> i32 {
    c * x
}

#[inline(always)]
fn clamp_u8(val: i32) -> u8 {
    if val <= 0 {
        0
    } else if val >= 255 {
        255
    } else {
        val as u8
    }
}

/// Compute 2D IDCT on an 8x8 block of dequantized coefficients.
/// Input: 64 i32 coefficients.
/// Output: 64 u8 pixels with range [0, 255] (level shift +128 applied).
pub fn idct_8x8(input: &[i32; 64], output: &mut [u8; 64]) {
    let mut workspace = [0i32; 64];

    // Pass 1: process rows
    for row in 0..8 {
        let offset = row * 8;
        let in0 = input[offset];
        let in1 = input[offset + 1];
        let in2 = input[offset + 2];
        let in3 = input[offset + 3];
        let in4 = input[offset + 4];
        let in5 = input[offset + 5];
        let in6 = input[offset + 6];
        let in7 = input[offset + 7];

        // Check for all AC coefficients zero in this row (DC-only fast path)
        if (in1 | in2 | in3 | in4 | in5 | in6 | in7) == 0 {
            let dc_val = in0 << PASS1_BITS;
            workspace[offset] = dc_val;
            workspace[offset + 1] = dc_val;
            workspace[offset + 2] = dc_val;
            workspace[offset + 3] = dc_val;
            workspace[offset + 4] = dc_val;
            workspace[offset + 5] = dc_val;
            workspace[offset + 6] = dc_val;
            workspace[offset + 7] = dc_val;
            continue;
        }

        // Even part
        let z2 = in2;
        let z3 = in6;
        let z1 = multiply(z2 + z3, FIX_0_541196100);
        let tmp2 = z1 + multiply(z3, -FIX_1_847759065);
        let tmp3 = z1 + multiply(z2, FIX_0_765366865);

        let tmp0 = (in0 + in4) << CONST_BITS;
        let tmp1 = (in0 - in4) << CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        // Odd part
        let mut tmp0 = in7;
        let mut tmp1 = in5;
        let mut tmp2 = in3;
        let mut tmp3 = in1;

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = multiply(z3 + z4, FIX_1_175875602);

        tmp0 = multiply(tmp0, FIX_0_298631336);
        tmp1 = multiply(tmp1, FIX_2_053119869);
        tmp2 = multiply(tmp2, FIX_3_072711026);
        tmp3 = multiply(tmp3, FIX_1_501321110);
        let z1 = multiply(z1, -FIX_0_899976223);
        let z2 = multiply(z2, -FIX_2_562915447);
        let z3 = multiply(z3, -FIX_1_961570560);
        let z4 = multiply(z4, -FIX_0_390180644);

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        // Final row output with rounding
        workspace[offset] = (tmp10 + tmp3 + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 7] = (tmp10 - tmp3 + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 1] = (tmp11 + tmp2 + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 6] = (tmp11 - tmp2 + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 2] = (tmp12 + tmp1 + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 5] = (tmp12 - tmp1 + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 3] = (tmp13 + tmp0 + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 4] = (tmp13 - tmp0 + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns
    for col in 0..8 {
        let in0 = workspace[col];
        let in1 = workspace[col + 8];
        let in2 = workspace[col + 16];
        let in3 = workspace[col + 24];
        let in4 = workspace[col + 32];
        let in5 = workspace[col + 40];
        let in6 = workspace[col + 48];
        let in7 = workspace[col + 56];

        // Even part
        let z2 = in2;
        let z3 = in6;
        let z1 = multiply(z2 + z3, FIX_0_541196100);
        let tmp2 = z1 + multiply(z3, -FIX_1_847759065);
        let tmp3 = z1 + multiply(z2, FIX_0_765366865);

        let tmp0 = (in0 + in4) << CONST_BITS;
        let tmp1 = (in0 - in4) << CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        // Odd part
        let mut tmp0 = in7;
        let mut tmp1 = in5;
        let mut tmp2 = in3;
        let mut tmp3 = in1;

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = multiply(z3 + z4, FIX_1_175875602);

        tmp0 = multiply(tmp0, FIX_0_298631336);
        tmp1 = multiply(tmp1, FIX_2_053119869);
        tmp2 = multiply(tmp2, FIX_3_072711026);
        tmp3 = multiply(tmp3, FIX_1_501321110);
        let z1 = multiply(z1, -FIX_0_899976223);
        let z2 = multiply(z2, -FIX_2_562915447);
        let z3 = multiply(z3, -FIX_1_961570560);
        let z4 = multiply(z4, -FIX_0_390180644);

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        // Final column output with level shift +128 and clamping
        const SHIFT: i32 = CONST_BITS + PASS1_BITS + 3;
        const ROUND: i32 = 1 << (SHIFT - 1);

        output[col] = clamp_u8(((tmp10 + tmp3 + ROUND) >> SHIFT) + 128);
        output[col + 56] = clamp_u8(((tmp10 - tmp3 + ROUND) >> SHIFT) + 128);
        output[col + 8] = clamp_u8(((tmp11 + tmp2 + ROUND) >> SHIFT) + 128);
        output[col + 48] = clamp_u8(((tmp11 - tmp2 + ROUND) >> SHIFT) + 128);
        output[col + 16] = clamp_u8(((tmp12 + tmp1 + ROUND) >> SHIFT) + 128);
        output[col + 40] = clamp_u8(((tmp12 - tmp1 + ROUND) >> SHIFT) + 128);
        output[col + 24] = clamp_u8(((tmp13 + tmp0 + ROUND) >> SHIFT) + 128);
        output[col + 32] = clamp_u8(((tmp13 - tmp0 + ROUND) >> SHIFT) + 128);
    }
}

/// Ultra-fast DC-only 2D IDCT (for blocks where all 63 AC coefficients are zero)
#[inline(always)]
pub fn idct_8x8_dc(dc_coeff: i32, output: &mut [u8; 64]) {
    let val = clamp_u8(((dc_coeff + 4) >> 3) + 128);
    output.fill(val);
}

/// Direct strided 2D IDCT writing directly into MCU or image buffer
#[inline(always)]
pub fn idct_8x8_stride(input: &[i32; 64], output: &mut [u8], start: usize, stride: usize) {
    let mut workspace = [0i32; 64];

    // Pass 1: process rows
    for row in 0..8 {
        let offset = row * 8;
        let in0 = input[offset];
        let in1 = input[offset + 1];
        let in2 = input[offset + 2];
        let in3 = input[offset + 3];
        let in4 = input[offset + 4];
        let in5 = input[offset + 5];
        let in6 = input[offset + 6];
        let in7 = input[offset + 7];

        if (in1 | in2 | in3 | in4 | in5 | in6 | in7) == 0 {
            let dc_val = in0 << PASS1_BITS;
            workspace[offset..offset + 8].fill(dc_val);
            continue;
        }

        // Even part
        let z2 = in2;
        let z3 = in6;
        let z1 = multiply(z2 + z3, FIX_0_541196100);
        let tmp2 = z1 + multiply(z3, -FIX_1_847759065);
        let tmp3 = z1 + multiply(z2, FIX_0_765366865);

        let tmp0 = (in0 + in4) << CONST_BITS;
        let tmp1 = (in0 - in4) << CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        // Odd part
        let mut tmp0 = in7;
        let mut tmp1 = in5;
        let mut tmp2 = in3;
        let mut tmp3 = in1;

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = multiply(z3 + z4, FIX_1_175875602);

        tmp0 = multiply(tmp0, FIX_0_298631336);
        tmp1 = multiply(tmp1, FIX_2_053119869);
        tmp2 = multiply(tmp2, FIX_3_072711026);
        tmp3 = multiply(tmp3, FIX_1_501321110);
        let z1 = multiply(z1, -FIX_0_899976223);
        let z2 = multiply(z2, -FIX_2_562915447);
        let z3 = multiply(z3, -FIX_1_961570560);
        let z4 = multiply(z4, -FIX_0_390180644);

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        workspace[offset] = (tmp10 + tmp3) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 7] = (tmp10 - tmp3) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 1] = (tmp11 + tmp2) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 6] = (tmp11 - tmp2) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 2] = (tmp12 + tmp1) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 5] = (tmp12 - tmp1) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 3] = (tmp13 + tmp0) >> (CONST_BITS - PASS1_BITS);
        workspace[offset + 4] = (tmp13 - tmp0) >> (CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns
    for col in 0..8 {
        let in0 = workspace[col];
        let in1 = workspace[col + 8];
        let in2 = workspace[col + 16];
        let in3 = workspace[col + 24];
        let in4 = workspace[col + 32];
        let in5 = workspace[col + 40];
        let in6 = workspace[col + 48];
        let in7 = workspace[col + 56];

        // Even part
        let z2 = in2;
        let z3 = in6;
        let z1 = multiply(z2 + z3, FIX_0_541196100);
        let tmp2 = z1 + multiply(z3, -FIX_1_847759065);
        let tmp3 = z1 + multiply(z2, FIX_0_765366865);

        let tmp0 = (in0 + in4) << CONST_BITS;
        let tmp1 = (in0 - in4) << CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        // Odd part
        let mut tmp0 = in7;
        let mut tmp1 = in5;
        let mut tmp2 = in3;
        let mut tmp3 = in1;

        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = multiply(z3 + z4, FIX_1_175875602);

        tmp0 = multiply(tmp0, FIX_0_298631336);
        tmp1 = multiply(tmp1, FIX_2_053119869);
        tmp2 = multiply(tmp2, FIX_3_072711026);
        tmp3 = multiply(tmp3, FIX_1_501321110);
        let z1 = multiply(z1, -FIX_0_899976223);
        let z2 = multiply(z2, -FIX_2_562915447);
        let z3 = multiply(z3, -FIX_1_961570560);
        let z4 = multiply(z4, -FIX_0_390180644);

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        const SHIFT: i32 = CONST_BITS + PASS1_BITS + 3;
        const ROUND: i32 = 1 << (SHIFT - 1);

        output[start + col] = clamp_u8(((tmp10 + tmp3 + ROUND) >> SHIFT) + 128);
        output[start + stride * 7 + col] = clamp_u8(((tmp10 - tmp3 + ROUND) >> SHIFT) + 128);
        output[start + stride + col] = clamp_u8(((tmp11 + tmp2 + ROUND) >> SHIFT) + 128);
        output[start + stride * 6 + col] = clamp_u8(((tmp11 - tmp2 + ROUND) >> SHIFT) + 128);
        output[start + stride * 2 + col] = clamp_u8(((tmp12 + tmp1 + ROUND) >> SHIFT) + 128);
        output[start + stride * 5 + col] = clamp_u8(((tmp12 - tmp1 + ROUND) >> SHIFT) + 128);
        output[start + stride * 3 + col] = clamp_u8(((tmp13 + tmp0 + ROUND) >> SHIFT) + 128);
        output[start + stride * 4 + col] = clamp_u8(((tmp13 - tmp0 + ROUND) >> SHIFT) + 128);
    }
}

/// Direct strided DC-only IDCT
#[inline(always)]
pub fn idct_8x8_dc_stride(dc_coeff: i32, output: &mut [u8], start: usize, stride: usize) {
    let val = clamp_u8(((dc_coeff + 4) >> 3) + 128);
    for row in 0..8 {
        let offset = start + row * stride;
        output[offset..offset + 8].fill(val);
    }
}
