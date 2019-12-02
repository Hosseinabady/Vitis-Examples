#include "sgemv.hpp"
#include <hls_stream.h>




extern "C" {


	void dot_product(
		DATA_TYPE *A,
		DATA_TYPE *x,
		DATA_TYPE *y,
		u32        n,
		u32        m,
		DATA_TYPE  alpha,
		DATA_TYPE  beta
	) {
#pragma HLS DATAFLOW

		hls::stream<DATA_TYPE>       a_fifo;
		hls::stream<DATA_TYPE>       y_fifo;


		DATA_TYPE                    sum = 0;
		for (u32 i = 0; i < n; i++) {
			for (u32 j = 0; j < m; j++) {
#pragma HLS PIPELINE
				a_fifo << A[i*m+j];
			}
		}

		for (u32 i = 0; i < n; i++) {
			for (u32 j = 0; j < m; j++) {
#pragma HLS PIPELINE
				if (j == 0)
					sum = 0;
				DATA_TYPE a = a_fifo.read();
				sum += a*x[j];
				if(j == m-1)
					y_fifo << sum;
			}
		}

		for (u32 i = 0; i < n; i++) {
#pragma HLS PIPELINE
			y[i] = alpha*y_fifo.read() + beta*y[i];
		}

	}

	void sgemv(
		DATA_TYPE *A,
		DATA_TYPE *x,
		DATA_TYPE *y,
		u32        n,
		u32        m,
		DATA_TYPE  alpha,
		DATA_TYPE  beta
	) {
#pragma HLS INTERFACE m_axi     port=A  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=x  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=y  offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=A               bundle=control
#pragma HLS INTERFACE s_axilite port=x               bundle=control
#pragma HLS INTERFACE s_axilite port=y               bundle=control

#pragma HLS INTERFACE s_axilite port=n               bundle=control
#pragma HLS INTERFACE s_axilite port=m               bundle=control
#pragma HLS INTERFACE s_axilite port=alpha           bundle=control
#pragma HLS INTERFACE s_axilite port=beta            bundle=control

#pragma HLS INTERFACE s_axilite port=return          bundle=control

		DATA_TYPE x_local[COL_SIZE_MAX];
		DATA_TYPE y_local[ROW_SIZE_MAX];

		for (u32 i = 0; i < m; i++) {
#pragma HLS PIPELINE
			x_local[i] = x[i];
		}

		for (u32 i = 0; i < n; i++) {
#pragma HLS PIPELINE
			y_local[i] = y[i];
		}


		dot_product(A, x_local, y_local, n, m, alpha, beta );

		for (u32 i = 0; i < n; i++) {
#pragma HLS PIPELINE
			y[i] = y_local[i];
		}


	}

}
