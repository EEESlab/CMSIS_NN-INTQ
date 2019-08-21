 /*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 * Modifications Copyright (C) 2018 University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library - Mixed Precision INT-Q
 * Title:        arm_u2_to_int16_reordered.c
 * Description:  Converts the elements of u2 vector to
 *               a reordered int16 vector (without left-shift).
 *
 * $Date:        March 2019
 * $Authors:     Alessandro Capotondi - alessandro.capotondi@unibo.it
 *               Manuele Rusci - manuele.rusci@unibo.it
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup nndata_convert
 * @{
 */

/**
 * @brief Converts the elements of u2 vector to
 *        a reordered int16 vector (without left-shift).
 * @param[in] *pSrc       points to the u2 input vector
 * @param[out] *pDst      points to the int16 output vector
 * @param[in] blockSize   length of the input vector
 * @param[in] offset      input quantization offset
 * @return none.
 */

void arm_u2_to_int16_reordered(
		const uint8_t *pSrc,
		int16_t *pDst,
		uint32_t blockSize,
	    const uint8_t offset)
{
    const uint8_t *pIn = pSrc;  /* Src pointer */
    uint32_t  blkCnt;           /* loop counter */
    int16_t offsets[2] = {offset, offset};
    const int16_t *offset_ptr = offsets;

#ifndef ARM_MATH_CM0_FAMILY
    int32_t   in;
    int32_t   in1, in2, in3, in4, in5, in6, in7, in8;
    int32_t   offset_vect = *__SIMD32(offset_ptr);

    /*loop Unrolling */
    blkCnt = blockSize >> 4u; // 16-elements block

    /* First part of the processing with loop unrolling.
       Second loop below computes the leftover */
    if(offset)
    {
		while (blkCnt > 0u)
		{
			in = *__SIMD32(pIn)++;

            in8 = __UXTB16(__ROR(__UXTB16( in << 6), 6));
            in7 = __UXTB16(__ROR(__UXTB16( in << 4), 6));
            in6 = __UXTB16(__ROR(__UXTB16( in << 2), 6));
            in5 = __UXTB16(__ROR(__UXTB16( in     ), 6));

            in4 = __UXTB16(__ROR(__UXTB16( in >> 2), 6));
            in3 = __UXTB16(__ROR(__UXTB16( in >> 4), 6));
            in2 = __UXTB16(__ROR(__UXTB16( in >> 6), 6));
            in1 = __UXTB16(__ROR(__UXTB16( in >> 8), 6));

	        in8 = __SSUB16(in8, offset_vect);
	        in7 = __SSUB16(in7, offset_vect);
	        in6 = __SSUB16(in6, offset_vect);
	        in5 = __SSUB16(in5, offset_vect);
            in4 = __SSUB16(in4, offset_vect);
	        in3 = __SSUB16(in3, offset_vect);
            in2 = __SSUB16(in2, offset_vect);
	        in1 = __SSUB16(in1, offset_vect);

#ifndef ARM_MATH_BIG_ENDIAN
            *__SIMD32(pDst)++ = in8;
            *__SIMD32(pDst)++ = in7;
            *__SIMD32(pDst)++ = in6;
            *__SIMD32(pDst)++ = in5;
            *__SIMD32(pDst)++ = in4;
            *__SIMD32(pDst)++ = in3;
            *__SIMD32(pDst)++ = in2;
            *__SIMD32(pDst)++ = in1;
#else
            *__SIMD32(pDst)++ = in1;
            *__SIMD32(pDst)++ = in2;
            *__SIMD32(pDst)++ = in3;
            *__SIMD32(pDst)++ = in4;
            *__SIMD32(pDst)++ = in5;
            *__SIMD32(pDst)++ = in6;
            *__SIMD32(pDst)++ = in7;
            *__SIMD32(pDst)++ = in8;
#endif

			/* Decrement the loop counter */
			blkCnt--;
		}
    }
    else
    {
		while (blkCnt > 0u)
		{
			in = *__SIMD32(pIn)++;

            in8 = __UXTB16(__ROR(__UXTB16( in << 6), 6));
            in7 = __UXTB16(__ROR(__UXTB16( in << 4), 6));
            in6 = __UXTB16(__ROR(__UXTB16( in << 2), 6));
            in5 = __UXTB16(__ROR(__UXTB16( in     ), 6));

            in4 = __UXTB16(__ROR(__UXTB16( in >> 2), 6));
            in3 = __UXTB16(__ROR(__UXTB16( in >> 4), 6));
            in2 = __UXTB16(__ROR(__UXTB16( in >> 6), 6));
            in1 = __UXTB16(__ROR(__UXTB16( in >> 8), 6));

#ifndef ARM_MATH_BIG_ENDIAN
            *__SIMD32(pDst)++ = in8;
            *__SIMD32(pDst)++ = in7;
            *__SIMD32(pDst)++ = in6;
            *__SIMD32(pDst)++ = in5;
            *__SIMD32(pDst)++ = in4;
            *__SIMD32(pDst)++ = in3;
            *__SIMD32(pDst)++ = in2;
            *__SIMD32(pDst)++ = in1;
#else
            *__SIMD32(pDst)++ = in1;
            *__SIMD32(pDst)++ = in2;
            *__SIMD32(pDst)++ = in3;
            *__SIMD32(pDst)++ = in4;
            *__SIMD32(pDst)++ = in5;
            *__SIMD32(pDst)++ = in6;
            *__SIMD32(pDst)++ = in7;
            *__SIMD32(pDst)++ = in8;
#endif

			/* Decrement the loop counter */
			blkCnt--;
		}
    }

    /* If the blockSize is not a multiple, compute any remaining output samples here.
     ** No loop unrolling is used. */
    blkCnt = blockSize % 0x16u;
#else
#error "Cortex-M0 is not supported"
#endif /* #ifndef ARM_MATH_CM0_FAMILY */

    int32_t in_32 = *((int32_t *)  pIn);

    while (blkCnt > 0u)
    {
        *pDst++ = (int16_t) __UXTB16(__ROR(__UXTB16( in_32 << 6), 6));
        in_32 = __ROR(in_32, 2);

        /* Decrement the loop counter */
        blkCnt--;
    }

}

/**
 * @}
 */