#include "convolution_kernel.h"




extern "C" {


void convolution_accel(
		data_type *input,
		data_type *output,
		data_type *mask,
		int        n,
		int        m,

		int        p,
		int        q
)
{
#pragma HLS INTERFACE s_axilite port=return  bundle=control

#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=m bundle=control
#pragma HLS INTERFACE s_axilite port=p bundle=control
#pragma HLS INTERFACE s_axilite port=q bundle=control

#pragma HLS INTERFACE m_axi depth=256 port=input  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi depth=256 port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi depth=256 port=mask   offset=slave bundle=gmem2


	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			data_type out_data = 0;
			for (int h = 0; h < p; h++) {
				for (int w = 0; w < q; w++) {
					if ( (i-p/2+h >= 0 && i-p/2+h < n) &&
						 (j-q/2+w >= 0 && j-q/2+w < m)
						)
						out_data += input[(i-p/2+h)*m+j-q/2+w]*mask[h*q+w];
				}
			}
			output[i*m+j] = out_data;
		}
	}

}

}
