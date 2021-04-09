/*
 * Ported from mozjpeg to rust
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2015, 2020, D. R. Commander.
 *
 * Conditions of distribution and use:
 * In plain English:
 *
 * 1. We don't promise that this software works.  (But if you find any bugs,
 *    please let us know!)
 * 2. You can use this software for whatever you want.  You don't have to pay us.
 * 3. You may not pretend that you wrote this software.  If you use it in a
 *    program, you must acknowledge somewhere in your documentation that
 *    you've used the IJG code.
 *
 * In legalese:
 *
 * The authors make NO WARRANTY or representation, either express or implied,
 * with respect to this software, its quality, accuracy, merchantability, or
 * fitness for a particular purpose.  This software is provided "AS IS", and you,
 * its user, assume the entire risk as to its quality and accuracy.
 *
 * This software is copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
 * All Rights Reserved except as specified below.
 *
 * Permission is hereby granted to use, copy, modify, and distribute this
 * software (or portions thereof) for any purpose, without fee, subject to these
 * conditions:
 * (1) If any part of the source code for this software is distributed, then this
 * README file must be included, with this copyright and no-warranty notice
 * unaltered; and any additions, deletions, or changes to the original files
 * must be clearly indicated in accompanying documentation.
 * (2) If only executable code is distributed, then the accompanying
 * documentation must state that "this software is based in part on the work of
 * the Independent JPEG Group".
 * (3) Permission for use of this software is granted only if the user accepts
 * full responsibility for any undesirable consequences; the authors accept
 * NO LIABILITY for damages of any kind.
 *
 * These conditions apply to any software derived from or based on the IJG code,
 * not just to the unmodified library.  If you use our work, you ought to
 * acknowledge us.
 *
 * Permission is NOT granted for the use of any IJG author's name or company name
 * in advertising or publicity relating to this software or products derived from
 * it.  This software may be referred to only as "the Independent JPEG Group's
 * software".
 *
 * We specifically permit and encourage the use of this software as the basis of
 * commercial products, provided that all warranty or liability claims are
 * assumed by the product vendor.
 *
 * This file contains a slower but more accurate integer implementation of the
 * forward DCT (Discrete Cosine Transform).
 *
 * A 2-D DCT can be done by 1-D DCT on each row followed by 1-D DCT
 * on each column.  Direct algorithms are also available, but they are
 * much more complex and seem not to be any faster when reduced to code.
 *
 * This implementation is based on an algorithm described in
 *   C. Loeffler, A. Ligtenberg and G. Moschytz, "Practical Fast 1-D DCT
 *   Algorithms with 11 Multiplications", Proc. Int'l. Conf. on Acoustics,
 *   Speech, and Signal Processing 1989 (ICASSP '89), pp. 988-991.
 * The primary algorithm described there uses 11 multiplies and 29 adds.
 * We use their alternate method with 12 multiplies and 32 adds.
 * The advantage of this method is that no data path contains more than one
 * multiplication; this allows a very simple and accurate implementation in
 * scaled fixed-point arithmetic, with a minimal number of shifts.
 */

static CONST_BITS: i32 = 13;
static PASS1_BITS: i32 = 2;

static FIX_0_298631336: i32 = 2446;
static FIX_0_390180644: i32 = 3196;
static FIX_0_541196100: i32 = 4433;
static FIX_0_765366865: i32 = 6270;
static FIX_0_899976223: i32 = 7373;
static FIX_1_175875602: i32 = 9633;
static FIX_1_501321110: i32 = 12299;
static FIX_1_847759065: i32 = 15137;
static FIX_1_961570560: i32 = 16069;
static FIX_2_053119869: i32 = 16819;
static FIX_2_562915447: i32 = 20995;
static FIX_3_072711026: i32 = 25172;

const DCT_SIZE: usize = 8;

#[inline(always)]
fn descale(x: i32, n: i32) -> i32 {
    x >> n
}

#[inline(always)]
fn into_el(v: i32) -> i16 {
    v as i16
}

#[allow(clippy::erasing_op)]
#[allow(clippy::identity_op)]
pub(crate) fn fdct(data: &mut [i16; 64]) {
    /* Pass 1: process rows. */
    /* Note results are scaled up by sqrt(8) compared to a true DCT; */
    /* furthermore, we scale the results by 2**PASS1_BITS. */

    for y in 0..8 {
        let offset = y * 8;

        let tmp0 = i32::from(data[offset + 0]) + i32::from(data[offset + 7]);
        let tmp7 = i32::from(data[offset + 0]) - i32::from(data[offset + 7]);
        let tmp1 = i32::from(data[offset + 1]) + i32::from(data[offset + 6]);
        let tmp6 = i32::from(data[offset + 1]) - i32::from(data[offset + 6]);
        let tmp2 = i32::from(data[offset + 2]) + i32::from(data[offset + 5]);
        let tmp5 = i32::from(data[offset + 2]) - i32::from(data[offset + 5]);
        let tmp3 = i32::from(data[offset + 3]) + i32::from(data[offset + 4]);
        let tmp4 = i32::from(data[offset + 3]) - i32::from(data[offset + 4]);

        /* Even part per LL&M figure 1 --- note that published figure is faulty;
         * rotator "sqrt(2)*c1" should be "sqrt(2)*c6".
         */

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[offset + 0] = into_el((tmp10 + tmp11) << PASS1_BITS);
        data[offset + 4] = into_el((tmp10 - tmp11) << PASS1_BITS);

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[offset + 2] = into_el(descale(z1 + (tmp13 * FIX_0_765366865), CONST_BITS - PASS1_BITS));
        data[offset + 6] = into_el(descale(z1 + (tmp12 * -FIX_1_847759065), CONST_BITS - PASS1_BITS));

        /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
         * cK represents cos(K*pi/16).
         * i0..i3 in the paper are tmp4..tmp7 here.
         */

        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602; /* sqrt(2) * c3 */

        let tmp4 = tmp4 * FIX_0_298631336; /* sqrt(2) * (-c1+c3+c5-c7) */
        let tmp5 = tmp5 * FIX_2_053119869; /* sqrt(2) * ( c1+c3-c5+c7) */
        let tmp6 = tmp6 * FIX_3_072711026; /* sqrt(2) * ( c1+c3+c5-c7) */
        let tmp7 = tmp7 * FIX_1_501321110; /* sqrt(2) * ( c1+c3-c5-c7) */
        let z1 = z1 * -FIX_0_899976223; /* sqrt(2) * ( c7-c3) */
        let z2 = z2 * -FIX_2_562915447; /* sqrt(2) * (-c1-c3) */
        let z3 = z3 * -FIX_1_961570560; /* sqrt(2) * (-c3-c5) */
        let z4 = z4 * -FIX_0_390180644; /* sqrt(2) * ( c5-c3) */

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        data[offset + 7] = into_el(descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS));
        data[offset + 5] = into_el(descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS));
        data[offset + 3] = into_el(descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS));
        data[offset + 1] = into_el(descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS));
    }

    /* Pass 2: process columns.
     * We remove the PASS1_BITS scaling, but leave the results scaled up
     * by an overall factor of 8.
     */

    for x in 0..8 {
        let tmp0 = i32::from(data[DCT_SIZE * 0 + x]) + i32::from(data[DCT_SIZE * 7 + x]);
        let tmp7 = i32::from(data[DCT_SIZE * 0 + x]) - i32::from(data[DCT_SIZE * 7 + x]);
        let tmp1 = i32::from(data[DCT_SIZE * 1 + x]) + i32::from(data[DCT_SIZE * 6 + x]);
        let tmp6 = i32::from(data[DCT_SIZE * 1 + x]) - i32::from(data[DCT_SIZE * 6 + x]);
        let tmp2 = i32::from(data[DCT_SIZE * 2 + x]) + i32::from(data[DCT_SIZE * 5 + x]);
        let tmp5 = i32::from(data[DCT_SIZE * 2 + x]) - i32::from(data[DCT_SIZE * 5 + x]);
        let tmp3 = i32::from(data[DCT_SIZE * 3 + x]) + i32::from(data[DCT_SIZE * 4 + x]);
        let tmp4 = i32::from(data[DCT_SIZE * 3 + x]) - i32::from(data[DCT_SIZE * 4 + x]);

        /* Even part per LL&M figure 1 --- note that published figure is faulty;
         * rotator "sqrt(2)*c1" should be "sqrt(2)*c6".
         */

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[DCT_SIZE * 0 + x] = into_el(descale(tmp10 + tmp11, PASS1_BITS));
        data[DCT_SIZE * 4 + x] = into_el(descale(tmp10 - tmp11, PASS1_BITS));

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[DCT_SIZE * 2 + x] = into_el(descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS));
        data[DCT_SIZE * 6 + x] = into_el(descale(z1 + tmp12 * -FIX_1_847759065, CONST_BITS + PASS1_BITS));

        /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
         * cK represents cos(K*pi/16).
         * i0..i3 in the paper are tmp4..tmp7 here.
         */

        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602; /* sqrt(2) * c3 */

        let tmp4 = tmp4 * FIX_0_298631336; /* sqrt(2) * (-c1+c3+c5-c7) */
        let tmp5 = tmp5 * FIX_2_053119869; /* sqrt(2) * ( c1+c3-c5+c7) */
        let tmp6 = tmp6 * FIX_3_072711026; /* sqrt(2) * ( c1+c3+c5-c7) */
        let tmp7 = tmp7 * FIX_1_501321110; /* sqrt(2) * ( c1+c3-c5-c7) */
        let z1 = z1 * -FIX_0_899976223; /* sqrt(2) * ( c7-c3) */
        let z2 = z2 * -FIX_2_562915447; /* sqrt(2) * (-c1-c3) */
        let z3 = z3 * -FIX_1_961570560; /* sqrt(2) * (-c3-c5) */
        let z4 = z4 * -FIX_0_390180644; /* sqrt(2) * ( c5-c3) */

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        data[DCT_SIZE * 7 + x] = into_el(descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS));
        data[DCT_SIZE * 5 + x] = into_el(descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS));
        data[DCT_SIZE * 3 + x] = into_el(descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS));
        data[DCT_SIZE * 1 + x] = into_el(descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS));
    }
}
