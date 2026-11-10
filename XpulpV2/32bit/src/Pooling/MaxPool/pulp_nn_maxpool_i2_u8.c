/*
 * pulp_nn_maxpool_i2_u8.c
 * Max pooling with mixed input/output precision and optional requantization
 * (out = clip((max * lambda + out_add) >> out_shift)), same API as pulp_nn_avgpool_new.t
 * by Georg Rutishauser <georgr@iis.ee.ethz.ch>, from which it is derived.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"




#define bitins(dst,not_mask_imm,src,mask_imm,off) __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)
#define bitext_u(x,size,off) __builtin_pulp_bextractu(x,size,off)
#define bitext(x,size,off) __builtin_pulp_bextract(x,size,off)

void __attribute__ ((noinline))  pulp_nn_maxpool_i2_u8(
  int8_t * pIn,
  uint8_t * pOut,
  int32_t lambda,
  uint16_t out_shift,
  int32_t out_add,
  uint16_t dim_im_in_x,
  uint16_t dim_im_in_y,
  uint16_t ch_im_in,
  uint16_t dim_im_out_x,
  uint16_t dim_im_out_y,
  uint16_t dim_kernel_x,
  uint16_t dim_kernel_y,
  uint16_t padding_t,
  uint16_t padding_b,
  uint16_t padding_l,
  uint16_t padding_r,
  uint16_t stride_x,
  uint16_t stride_y,
  int flag_requant
)
{
  /* parallelization */
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  if (dim_im_out_y < NUM_CORES)
  {
    n_cores = dim_im_out_y;
  }
  int Log2Core = log2(n_cores);
  int chunck = (dim_im_out_y >> Log2Core) + ((dim_im_out_y & (n_cores -1))!=0);
  int start = chunck * core_id;
  int stop = min(start + chunck, dim_im_out_y);
  int   i_x, i_y;


  int ch_im_in_r = ch_im_in >> 2;
  int ch_im_out_r = ch_im_in >> 0;
  int32_t maxv[4];
  for (i_y = start; i_y < stop; i_y++)
    {
        for (i_x = 0; i_x < dim_im_out_x; i_x++)
        {
            int k_y_start, k_y_end;
            int k_x_start, k_x_end;

            const int8_t *pTmp, *pTmpInner;
            uint8_t *pDst;

            // window rows/columns inside the input; padded positions never win the max
            k_y_start = maxs32(0, i_y * stride_y - padding_t);
            k_y_end = mins32(i_y * stride_y - padding_t + dim_kernel_y, dim_im_in_y);

            k_x_start = maxs32(0, i_x * stride_x - padding_l);
            k_x_end = mins32(i_x * stride_x - padding_l + dim_kernel_x, dim_im_in_x);

            pTmp = pIn;
            pDst = &pOut[ch_im_out_r * (i_x + i_y * dim_im_out_x)];
            int k_x, k_y;

            for (int ch_cnt = 0; ch_cnt < ch_im_in_r; ch_cnt++)
            {
              maxv[0] = -2;
              maxv[1] = -2;
              maxv[2] = -2;
              maxv[3] = -2;
              uint8_t out_el = 0;
                for (k_y = k_y_start; k_y < k_y_end; k_y++)
                {
                    for (k_x = k_x_start; k_x < k_x_end; k_x++)
                    {
                        pTmpInner = pTmp + (ch_im_in_r * (k_x + k_y * dim_im_in_x));
                        int8_t cur_chans = *pTmpInner;
                        maxv[0] = maxs32(maxv[0], (int32_t) bitext((int) cur_chans, 2, 0));
                        maxv[1] = maxs32(maxv[1], (int32_t) bitext((int) cur_chans, 2, 2));
                        maxv[2] = maxs32(maxv[2], (int32_t) bitext((int) cur_chans, 2, 4));
                        maxv[3] = maxs32(maxv[3], (int32_t) bitext((int) cur_chans, 2, 6));
                    }
                }
                int32_t out_large;
                out_large = flag_requant ? ((maxv[0] * lambda + out_add) >> out_shift) : maxv[0];
                out_el = clip8(out_large);
                pDst[(ch_cnt << (2)) + 0] = out_el;
                out_large = flag_requant ? ((maxv[1] * lambda + out_add) >> out_shift) : maxv[1];
                out_el = clip8(out_large);
                pDst[(ch_cnt << (2)) + 1] = out_el;
                out_large = flag_requant ? ((maxv[2] * lambda + out_add) >> out_shift) : maxv[2];
                out_el = clip8(out_large);
                pDst[(ch_cnt << (2)) + 2] = out_el;
                out_large = flag_requant ? ((maxv[3] * lambda + out_add) >> out_shift) : maxv[3];
                out_el = clip8(out_large);
                pDst[(ch_cnt << (2)) + 3] = out_el;
                pTmp++;
            }
        }
    }
 pi_cl_team_barrier(0);
}
