#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "flash.h"

__global__ void flash_attention(const float* Q, const float* K, const float* V, float* l, float* m, float* O, const int N, const int d, const int tc, const int tr, const int bc, const int br, const float softmax_scale){
	int tx = threadIdx.x;
	int batch = blockIdx.x; int head = blockIdx.y;

	// Computing offsets
	int matrix_offset = gridDim.y*batch*N*d + head*N*d;
	int vector_offset = gridDim.y*batch*N + head*N;

	//Defining SRAM
	extern __shared__ float shared[];
	int Q_tile_size = br*d;
	int KV_tile_size = bc*d;
	float* Q_i = shared;
	float* K_j = &shared[Q_tile_size];
	float* V_j = &shared[Q_tile_size + KV_tile_size];
	float* S_ij = &shared[Q_tile_size + 2*KV_tile_size];

	// Loop over T_c
	for (int j = 0; j < tc; j++){
		// Load K_j, V_j into shared memory
		for (int e = tx; e < KV_tile_size; e += br){
			int KV_row = j*bc + e/d;
			bool in_bounds = KV_row < N;
			K_j[e] = in_bounds ? K[matrix_offset + (KV_tile_size*j) + e]: 0.0f;
			V_j[e] = in_bounds ? V[matrix_offset + (KV_tile_size*j) + e]: 0.0f;
		}
		__syncthreads();

		int KV_valid = min(bc, N - j*bc);

		// Loop over T_r
		for (int i=0; i < tr; i++){
			int Q_row = i*br + tx;
			if (Q_row < N){
				// Load Q_i into shared memory, l and m into registers
				for (int x = 0; x < d; x++){
					Q_i[(tx*d) + x] = Q[matrix_offset + (Q_tile_size*i) + (tx*d) + x];
				}
				
				float row_max_prev = m[vector_offset + br*i + tx];
				float row_l_prev = l[vector_offset + br*i + tx];

			// Compute S_ij and row_max
			float row_max = -INFINITY;
			for (int y=0; y < bc; y++){
				float temp = 0.0f;
				if (y < KV_valid){
					for (int x = 0; x < d; x++){
						temp += Q_i[(tx*d) + x]*K_j[(y*d) + x];
					}
				temp *= softmax_scale;
				}
				else{
					temp = -INFINITY;
				}

				S_ij[(bc*tx) + y] = temp;

				if (temp > row_max)
					row_max = temp;
					
			}

			// Compute P and row_l
			float row_l = 0;
			for (int y=0; y < bc; y++){
				S_ij[(bc*tx) + y] = __expf(S_ij[(bc*tx) + y] - row_max);
				row_l += S_ij[(bc*tx) + y];
			}

			// Compute new m and l
			float row_max_new = max(row_max_prev, row_max);
			float row_l_new = (__expf(row_max_prev - row_max_new)*row_l_prev) + (__expf(row_max - row_max_new)*row_l);

			// Write to device memory
			for (int x=0; x<d; x++){
				float temp = 0;
				for (int y=0; y<bc; y++){
					temp += S_ij[(bc*tx) + y]*V_j[(y*d) + x];
				}
				int idx = matrix_offset + (Q_tile_size*i) + (tx*d) + x;
				O[idx] = (1/row_l_new)*((row_l_prev*__expf(row_max_prev - row_max_new)*O[idx]) + __expf(row_max - row_max_new)*temp);
			}

			m[vector_offset + (br*i) + tx] = row_max_new;
			l[vector_offset + (br*i) + tx] = row_l_new;
		}

		}
		__syncthreads();
	}
}

torch::Tensor forward_pass(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
	const int B = Q.size(0), nh = Q.size(1), N = Q.size(2), d = Q.size(3);
    int M;
    cudaDeviceGetAttribute(&M, cudaDevAttrMaxSharedMemoryPerBlock, 0);
	M = M/sizeof(float);
    const int bc = M/(4 * d); // Floor not ceil
    const int br = std::min(bc, d);
    float softmax_scale = 1.0f / sqrtf(d);

	torch::Tensor O = torch::zeros_like(Q);
	torch::Tensor l = torch::zeros({B, nh, N}, Q.options());
	torch::Tensor m = torch::full({B, nh, N}, -INFINITY, Q.options());

    const int tr = ceil((float)N / (float)br);
    const int tc = ceil((float)N / (float)bc);
    const int sram_size = (br * d + 2 * bc * d + br * bc) * sizeof(float);  // The sizes of Q_i, K_j, V_j, S_ij

    dim3 grid_dim(B, nh);
    dim3 block_dim(br);

    flash_attention<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>(), N, d, tc, tr, bc, br, softmax_scale);
    return O;
}
