#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_fp16.h>
#include <mma.h>
#include "flash.cuh"

using namespace nvcuda;

// FP16 WMMA tile shape (half inputs, float accumulation)
constexpr int WM = 16, WN = 16, WK = 16;

// Each warp owns 32 consecutive rows of the tile (two 16-row WMMA tiles), so br = 32*warps
__global__ void flash_attention(const float* Q, const float* K, const float* V, float* l, float* m, float* O, const int N, const int d, const int tc, const int tr, const int bc, const int br, const float softmax_scale){
	int tx = threadIdx.x;
	int r0 = (tx/32)*32; // First row of this warp's strip
	int batch = blockIdx.x; int head = blockIdx.y;

	// Computing offsets
	int matrix_offset = gridDim.y*batch*N*d + head*N*d;
	int vector_offset = gridDim.y*batch*N + head*N;

	//Defining SRAM: float scores first, then the half operand tiles (WMMA needs 32-byte aligned pointers)
	extern __shared__ __align__(32) unsigned char smem[];
	int Q_tile_size = br*d;
	int KV_tile_size = bc*d;
	float* S_ij = reinterpret_cast<float*>(smem);
	__half* Q_i = reinterpret_cast<__half*>(&S_ij[br*bc]);
	__half* K_j = &Q_i[Q_tile_size];
	__half* V_j = &K_j[KV_tile_size];
	__half* P_ij = &V_j[KV_tile_size];

	// Loop over T_c
	for (int j = 0; j < tc; j++){
		// Load K_j, V_j into shared memory
		for (int e = tx; e < KV_tile_size; e += br){
			int KV_row = j*bc + e/d;
			bool in_bounds = KV_row < N;
			K_j[e] = __float2half(in_bounds ? K[matrix_offset + (KV_tile_size*j) + e]: 0.0f);
			V_j[e] = __float2half(in_bounds ? V[matrix_offset + (KV_tile_size*j) + e]: 0.0f);
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
				Q_i[x*br + tx] = __float2half(row_valid ? Q[matrix_offset + (Q_tile_size*i) + (tx*d) + x] : 0.0f);
			}

			float row_max_prev = row_valid ? m[vector_offset + br*i + tx] : -INFINITY;
			float row_l_prev = row_valid ? l[vector_offset + br*i + tx] : 0.0f;
			__syncwarp();

			// Compute S_ij = Q_i*K_j^T on tensor cores
			for (int mt = r0; mt < r0 + 32; mt += WM){
				for (int n = 0; n < bc; n += WN){
					wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
					wmma::fill_fragment(acc, 0.0f);
					for (int k = 0; k < d; k += WK){
						wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::col_major> Q_frag;
						wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> K_frag;
						wmma::load_matrix_sync(Q_frag, &Q_i[k*br + mt], br);
						wmma::load_matrix_sync(K_frag, &K_j[n*d + k], d);
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

			// Compute P (as half for the next MMA) and row_l
			float row_l = 0;
			for (int y=0; y < bc; y++){
				float p = __expf(S_ij[(y*br) + tx] - row_max);
				P_ij[(y*br) + tx] = __float2half(p);
				row_l += p;
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

			// Compute P_ij*V_j on tensor cores, 16 columns of d at a time. Each chunk is staged
			// in the first 16 columns of S_ij (no longer needed) and then folded into O
			for (int n = 0; n < d; n += WN){
				for (int mt = r0; mt < r0 + 32; mt += WM){
					wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
					wmma::fill_fragment(acc, 0.0f);
					for (int k = 0; k < bc; k += WK){
						wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::col_major> P_frag;
						wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> V_frag;
						wmma::load_matrix_sync(P_frag, &P_ij[k*br + mt], br);
						wmma::load_matrix_sync(V_frag, &V_j[k*d + n], d);
						wmma::mma_sync(acc, P_frag, V_frag, acc);
					}
					wmma::store_matrix_sync(&S_ij[mt], acc, br, wmma::mem_col_major);
				}
				__syncwarp();

				if (row_valid){
					for (int c = 0; c < WN; c++){
						int idx = matrix_offset + (Q_tile_size*i) + (tx*d) + n + c;
						O[idx] = a*O[idx] + b*S_ij[c*br + tx];
					}
				}
				__syncwarp();
			}

			if (row_valid){
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
    int M; // Bytes
    cudaDeviceGetAttribute(&M, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    const int br = 64; // 2 warps of 32 rows each
    // Shared bytes are 6*br*bc + 2*br*d + 4*bc*d; take the largest multiple of 16 that fits, capped at 64
    const int bc = std::min(64, (M - 2*br*d)/(6*br + 4*d) / 16 * 16);
	TORCH_CHECK(bc >= 16, "d = ", d, " is too large to fit in shared memory");
    float softmax_scale = 1.0f / sqrtf(d);

	torch::Tensor O = torch::zeros_like(Q);
	torch::Tensor l = torch::zeros({B, nh, N}, Q.options());
	torch::Tensor m = torch::full({B, nh, N}, -INFINITY, Q.options());

    const int tr = ceil((float)N / (float)br);
    const int tc = ceil((float)N / (float)bc);
    // S_ij in float; Q_i, K_j, V_j and P_ij in half
    const int sram_size = br * bc * sizeof(float) + (br * d + 2 * bc * d + br * bc) * sizeof(__half);
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
