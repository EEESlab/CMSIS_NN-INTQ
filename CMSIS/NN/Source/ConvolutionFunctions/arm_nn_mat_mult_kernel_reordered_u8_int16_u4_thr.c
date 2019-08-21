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
 * Title:        arm_nn_mat_mult_kernel_reordered_u8_int16_u4_thr.c
 * Description:  Matrix-Multiplication function for
 *               u8 x int16_t convolution with reordered columns.
 *               Output is then quantized to u4 using thr
 *               config.folding technique.
 *
 * $Date:        March 2019
 * $Authors:     Alessandro Capotondi - alessandro.capotondi@unibo.it
 *               Manuele Rusci - manuele.rusci@unibo.it
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_math.h"

  /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using thr config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       thresholds  pointer to thresholds for quantization
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *arm_nn_mat_mult_kernel_reordered_u8_int16_u4_thr(const uint8_t * pA,
                                                  const int16_t * pInBuffer,
                                                  const uint16_t ch_im_out,
                                                  const uint16_t numCol_A,
                                                  const int32_t * bias,
												  uint8_t * pOut,
                                                  const uint8_t z_a,
                                                  const int16_t * thresholds)
{

#if defined (ARM_MATH_DSP)
    /* set up the second output pointers */
    uint8_t *pOut2 = pOut + (ch_im_out>>1); // config.out_data_t: u4 (2CHs per-Bytes)
    int     i;
    const int16_t *pB = pInBuffer;
    const int16_t *pB2 = pB + numCol_A;


    int16_t VzA[2] = {z_a,z_a};
	const int16_t *pzA = VzA;
	int32_t inzA = *__SIMD32(pzA);

    /* Pre-compute z_a offset over the inputs */
    int32_t z_a_offset  = 0;
	int32_t z_a_offset2 = 0;

    for (i = 0; i < numCol_A; i += 2) {
    	int32_t inB1 = *__SIMD32(pB)++;
    	int32_t inB2 = *__SIMD32(pB2)++;
    	z_a_offset = __SMLAD(inzA, inB1, z_a_offset);
    	z_a_offset2 = __SMLAD(inzA, inB2, z_a_offset2);
    }

    //leftover
    if (numCol_A & 0x1)
    {
        int16_t inB1 = *pB;
        int16_t inB2 = *pB2;
        z_a_offset += inB1*z_a;
        z_a_offset2 += inB2*z_a;
    }

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        /* setup pointers for B */
        pB = pInBuffer;
        pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const uint8_t *pA2 = pA + numCol_A;

        int32_t     sum =  bias[i] - z_a_offset;
        int32_t     sum2 = bias[i] - z_a_offset2;
        int32_t     sum3 = bias[i + 1] - z_a_offset;
        int32_t     sum4 = bias[i + 1] - z_a_offset2;

        uint16_t  colCnt = numCol_A >> 2;

        /* accumulate over the vector */
        while (colCnt)
        {
        	int32_t inA11, inA12, inA21, inA22;

        	int32_t inB1 = *__SIMD32(pB)++;
        	int32_t inB2 = *__SIMD32(pB2)++;

            pA = (uint8_t *) read_and_pad_reordered_u8((void *)pA, &inA11, &inA12);
            pA2 = (uint8_t *) read_and_pad_reordered_u8((void *)pA2, &inA21, &inA22);

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);
            sum3 = __SMLAD(inA21, inB1, sum3);
            sum4 = __SMLAD(inA21, inB2, sum4);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);
            sum3 = __SMLAD(inA22, inB1, sum3);
            sum4 = __SMLAD(inA22, inB2, sum4);

            colCnt--;
        } /* while over colCnt */

#if 0
        //FIXME 

        colCnt = numCol_A & 0x3;
           while (colCnt)
           {
               int16_t   inA1 = (int16_t)*pA++;
               int16_t   inB1 = *pB++;
               int16_t   inA2 = (int16_t)*pA2++;
               int16_t   inB2 = *pB2++;

               //inA1 = inA1 - VzA[0];
               //inA2 = inA2 - VzA[0];

               sum  += inA1 * inB1;
               sum2 += inA1 * inB2;
               sum3 += inA2 * inB1;
               sum4 += inA2 * inB2;
               colCnt--;
           }/* while over colCnt */
#endif

        /* Thresholds (u4 output)*/
        uint8_t qsum = __int16_to_u4((int16_t) sum, &thresholds[i<<4]);
		uint8_t qsum3 = __int16_to_u4((int16_t) sum3, &thresholds[(i+1)<<4]);
        uint8_t qsum2 = __int16_to_u4((int16_t) sum2, &thresholds[i<<4]);
		uint8_t qsum4 = __int16_to_u4((int16_t) sum4, &thresholds[(i+1)<<4]);

        *pOut++  = ( qsum & 0x0F ) | (( qsum3 << 4 ) & 0xF0 );
        *pOut2++ = ( qsum2 & 0x0F ) | (( qsum4 << 4 ) & 0xF0 );



        /* skip the row computed with A2 */
        pA += numCol_A;

    } /* for over ch_im_out */

    pOut += ch_im_out>>1; // config.out_data_t: u4 (2CH per-Bytes)
#else
	#error "Cortex-M0 and Cortex-M3 not supported"
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
#endif /* ARM_MATH_DSP */

    /* return the new output pointer with offset */
    return pOut;
}
