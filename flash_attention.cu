#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <mma.h>
#include "flash.cuh"

using namespace nvcuda;

// Round each fragment element to TF32 before feeding it to the tensor cores
template <typename Frag>
__device__ void to_tf32(Frag& f){
	for (int t = 0; t < f.num_elements; t++)
		f.x[t] = wmma::__float_to_tf32(f.x[t]);
}

// Each warp owns 32 consecutive rows of the tile (two 16-row WMMA tiles), so br = 32*warps
__global__ void flash_attention(const float* Q, const float* K, const float* V, float* l, float* m, float* O, const int N, const int d, const int tc, const int tr, const int bc, const int br, const float softmax_scale){
	int tx = threadIdx.x;
	int r0 = (tx/32)*32; // First row of this warp's strip
	int batch = blockIdx.x; int head = blockIdx.y;

	// Computing offsets
	int matrix_offset = gridDim.y*batch*N*d + head*N*d;
	int vector_offset = gridDim.y*batch*N + head*N;

	//Defining SRAM (WMMA needs 32-byte aligned pointers)
	extern __shared__ __align__(32) float shared[];
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
		for (int i=blockIdx.z; i < tr; i+=gridDim.z){
			int Q_row = i*br + tx;
			// Every lane must take part in the MMAs, so rows past N are zero-padded instead of skipped
			bool row_valid = Q_row < N;

			// Load Q_i into shared memory, l and m into registers
			for (int x = 0; x < d; x++){
				Q_i[x*br + tx] = row_valid ? Q[matrix_offset + (Q_tile_size*i) + (tx*d) + x] : 0.0f;
			}

			float row_max_prev = row_valid ? m[vector_offset + br*i + tx] : -INFINITY;
			float row_l_prev = row_valid ? l[vector_offset + br*i + tx] : 0.0f;
			__syncwarp();

			// Compute S_ij = Q_i*K_j^T on tensor cores
			for (int mt = r0; mt < r0 + 32; mt += 16){
				for (int n = 0; n < bc; n += 16){
					wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
					wmma::fill_fragment(acc, 0.0f);
					for (int k = 0; k < d; k += 8){
						wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> Q_frag;
						wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> K_frag;
						wmma::load_matrix_sync(Q_frag, &Q_i[k*br + mt], br);
						wmma::load_matrix_sync(K_frag, &K_j[n*d + k], d);
						to_tf32(Q_frag); to_tf32(K_frag);
						wmma::mma_sync(acc, Q_frag, K_frag, acc);
					}
					wmma::store_matrix_sync(&S_ij[n*br + mt], acc, br, wmma::mem_col_major);
				}
			}
			__syncwarp();

			// Scale, mask and compute row_max
			float row_max = -INFINITY;
			for (int y=0; y < bc; y++){
				float temp = (y < KV_valid) ? S_ij[(y*br) + tx]*softmax_scale : -INFINITY;
				S_ij[(y*br) + tx] = temp;

				if (temp > row_max)
					row_max = temp;
			}

			// Compute P and row_l
			float row_l = 0;
			for (int y=0; y < bc; y++){
				S_ij[(y*br) + tx] = __expf(S_ij[(y*br) + tx] - row_max);
				row_l += S_ij[(y*br) + tx];
			}

			// Compute new m and l
			float row_max_new = max(row_max_prev, row_max);
			float row_l_new = (__expf(row_max_prev - row_max_new)*row_l_prev) + (__expf(row_max - row_max_new)*row_l);

			// Write to device memory
			//We first precompute some things outside loop
			float inv_l = 1.f/row_l_new;
			float a = row_l_prev*__expf(row_max_prev - row_max_new)*inv_l;
			float b = __expf(row_max - row_max_new)*inv_l;
			__syncwarp();

			// Compute P_ij*V_j on tensor cores, storing the result in Q_i (no longer needed)
			for (int mt = r0; mt < r0 + 32; mt += 16){
				for (int n = 0; n < d; n += 16){
					wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
					wmma::fill_fragment(acc, 0.0f);
					for (int k = 0; k < bc; k += 8){
						wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> P_frag;
						wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> V_frag;
						wmma::load_matrix_sync(P_frag, &S_ij[k*br + mt], br);
						wmma::load_matrix_sync(V_frag, &V_j[k*d + n], d);
						to_tf32(P_frag); to_tf32(V_frag);
						wmma::mma_sync(acc, P_frag, V_frag, acc);
					}
					wmma::store_matrix_sync(&Q_i[n*br + mt], acc, br, wmma::mem_col_major);
				}
			}
			__syncwarp();

			if (row_valid){
				for (int x=0; x<d; x++){
					int idx = matrix_offset + (Q_tile_size*i) + (tx*d) + x;
					O[idx] = a*O[idx] + b*Q_i[x*br + tx];
				}

				m[vector_offset + (br*i) + tx] = row_max_new;
				l[vector_offset + (br*i) + tx] = row_l_new;
			}
		}
		__syncthreads();
	}
}

torch::Tensor forward_pass(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int num_splits){
	const int B = Q.size(0), nh = Q.size(1), N = Q.size(2), d = Q.size(3);
	TORCH_CHECK(d % 16 == 0, "d must be a multiple of 16 for the tensor-core kernel, got ", d);
    int M;
    cudaDeviceGetAttribute(&M, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

	M = M/sizeof(float);
    const int br = 64; // 2 warps of 32 rows each
    const int bc = std::min(64, (M - br*d)/(2*d + br) / 16 * 16); // Largest multiple of 16 that fits, capped at 64
	TORCH_CHECK(bc >= 16, "d = ", d, " is too large to fit in shared memory");
    float softmax_scale = 1.0f / sqrtf(d);

	torch::Tensor O = torch::zeros_like(Q);
	torch::Tensor l = torch::zeros({B, nh, N}, Q.options());
	torch::Tensor m = torch::full({B, nh, N}, -INFINITY, Q.options());

    const int tr = ceil((float)N / (float)br);
    const int tc = ceil((float)N / (float)bc);
    const int sram_size = (br * d + 2 * bc * d + br * bc) * sizeof(float);  // The sizes of Q_i, K_j, V_j, S_ij
	cudaFuncSetAttribute(flash_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, sram_size); // Needed above 48 KB
																
	if (num_splits <= 0){
		int num_sms, blocks_per_sm;
		cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, flash_attention, br, sram_size);

		const int target_blocks = num_sms * blocks_per_sm;   // one full wave
		num_splits = (target_blocks + B * nh - 1) / (B * nh);
	}
	num_splits = std::max(1, std::min(num_splits, tr));

    dim3 grid_dim(B, nh, num_splits);
    dim3 block_dim(br);

    flash_attention<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>(), N, d, tc, tr, bc, br, softmax_scale);
    return O;
}
