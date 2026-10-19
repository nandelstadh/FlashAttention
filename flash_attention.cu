#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <tuple>
#include <cuda_fp16.h>
#include <mma.h>
#include "flash.cuh"

using namespace nvcuda;

// FP16 WMMA tile shape (half inputs, float accumulation)
constexpr int WM = 16, WN = 16, WK = 16;
constexpr unsigned FULL_MASK = 0xffffffff;

// One block per Q tile (FlashAttention-2 loop order): the block loops over all K/V tiles and keeps
// the running max, running sum and output accumulator on chip, writing O to global memory once.
// Each warp owns 16 consecutive rows of the tile (one WMMA tile). Lanes l and l + 16 share a row and
// each handle half of its columns, so br = 16*warps and blockDim.x = 2*br
__global__ void flash_attention(const float* Q, const float* K, const float* V, float* O, const int N, const int d, const int tc, const int bc, const int br, const float softmax_scale){
	int tx = threadIdx.x;
	int lane = tx%32;
	int r0 = (tx/32)*16;       // First row of this warp's tile
	int row = r0 + lane%16;    // This thread's row
	int half_id = lane/16;     // Which half of the row's columns this thread handles
	// Q tiles of one head are consecutive blocks, so the blocks resident at a time share K/V in L2
	int i = blockIdx.x; int head = blockIdx.y; int batch = blockIdx.z;

	// Computing offsets
	int matrix_offset = (batch*gridDim.y + head)*N*d;
	int Q_tile_size = br*d;
	int KV_tile_size = bc*d;
	int tile_offset = matrix_offset + Q_tile_size*i; // First element of this block's Q and O rows
	int y0 = half_id*(bc/2), y1 = y0 + bc/2;         // This thread's score columns
	int x0 = half_id*(d/2), x1 = x0 + d/2;           // This thread's output columns

	//Defining SRAM: float scores and output first, then the half operand tiles (WMMA needs 32-byte aligned pointers)
	extern __shared__ __align__(32) unsigned char smem[];
	float* S_ij = reinterpret_cast<float*>(smem);
	float* O_i = &S_ij[br*bc];
	__half* Q_i = reinterpret_cast<__half*>(&O_i[Q_tile_size]);
	__half* K_j = &Q_i[Q_tile_size];
	__half* V_j = &K_j[KV_tile_size];
	__half* P_ij = &V_j[KV_tile_size];

	// Load Q_i once (coalesced: consecutive threads read consecutive elements) and clear O_i.
	// Rows past N are zero-padded, since every lane must take part in the MMAs
	for (int e = tx; e < Q_tile_size; e += blockDim.x){
		int r = e/d, x = e%d;
		Q_i[x*br + r] = __float2half(i*br + r < N ? Q[tile_offset + e] : 0.0f);
		O_i[x*br + r] = 0.0f;
	}

	// Running max and sum of this thread's row (identical in both lanes of the row)
	float row_max = -INFINITY;
	float row_l = 0.0f;

	// Loop over T_c
	for (int j = 0; j < tc; j++){
		// Load K_j, V_j into shared memory
		for (int e = tx; e < KV_tile_size; e += blockDim.x){
			int KV_row = j*bc + e/d;
			bool in_bounds = KV_row < N;
			K_j[e] = __float2half(in_bounds ? K[matrix_offset + (KV_tile_size*j) + e]: 0.0f);
			V_j[e] = __float2half(in_bounds ? V[matrix_offset + (KV_tile_size*j) + e]: 0.0f);
		}
		__syncthreads();

		int KV_valid = min(bc, N - j*bc);

		// Compute this warp's rows of S_ij = Q_i*K_j^T on tensor cores
		for (int n = 0; n < bc; n += WN){
			wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
			wmma::fill_fragment(acc, 0.0f);
			for (int k = 0; k < d; k += WK){
				wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::col_major> Q_frag;
				wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> K_frag;
				wmma::load_matrix_sync(Q_frag, &Q_i[k*br + r0], br);
				wmma::load_matrix_sync(K_frag, &K_j[n*d + k], d);
				wmma::mma_sync(acc, Q_frag, K_frag, acc);
			}
			wmma::store_matrix_sync(&S_ij[n*br + r0], acc, br, wmma::mem_col_major);
		}
		__syncwarp();

		// Scale, mask the padding past N and compute the new row max over both halves
		float row_max_new = row_max;
		for (int y = y0; y < y1; y++){
			float temp = (y < KV_valid) ? S_ij[(y*br) + row]*softmax_scale : -INFINITY;
			S_ij[(y*br) + row] = temp;
			row_max_new = max(row_max_new, temp);
		}
		row_max_new = max(row_max_new, __shfl_xor_sync(FULL_MASK, row_max_new, 16));

		// Compute P (as half for the next MMA) and update the running sum
		float row_sum = 0.0f;
		for (int y = y0; y < y1; y++){
			float p = __expf(S_ij[(y*br) + row] - row_max_new);
			P_ij[(y*br) + row] = __float2half(p);
			row_sum += p;
		}
		row_sum += __shfl_xor_sync(FULL_MASK, row_sum, 16);
		float alpha = __expf(row_max - row_max_new); // Rescales everything accumulated so far
		row_l = alpha*row_l + row_sum;
		row_max = row_max_new;

		// Rescale this thread's half of the row of the output accumulator
		for (int x = x0; x < x1; x++){
			O_i[x*br + row] *= alpha;
		}
		__syncwarp();

		// O_i += P_ij*V_j on tensor cores, accumulating straight into the shared output tile
		for (int n = 0; n < d; n += WN){
			wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
			wmma::load_matrix_sync(acc, &O_i[n*br + r0], br, wmma::mem_col_major);
			for (int k = 0; k < bc; k += WK){
				wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::col_major> P_frag;
				wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> V_frag;
				wmma::load_matrix_sync(P_frag, &P_ij[k*br + r0], br);
				wmma::load_matrix_sync(V_frag, &V_j[k*d + n], d);
				wmma::mma_sync(acc, P_frag, V_frag, acc);
			}
			wmma::store_matrix_sync(&O_i[n*br + r0], acc, br, wmma::mem_col_major);
		}
		// K_j and V_j are overwritten by the next iteration
		__syncthreads();
	}

	// Normalise this thread's half of the row, then write O once (coalesced)
	float inv_l = 1.f/row_l;
	for (int x = x0; x < x1; x++){
		O_i[x*br + row] *= inv_l;
	}
	__syncthreads();
	for (int e = tx; e < Q_tile_size; e += blockDim.x){
		int r = e/d, x = e%d;
		if (i*br + r < N)
			O[tile_offset + e] = O_i[x*br + r];
	}
}

// Shared bytes for a br x bc tile: S_ij and O_i in float; Q_i, K_j, V_j and P_ij in half
static int sram_bytes(int br, int bc, int d){
	return (br*bc + br*d)*sizeof(float) + (br*d + 2*bc*d + br*bc)*sizeof(__half);
}

torch::Tensor forward_pass(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
	const int B = Q.size(0), nh = Q.size(1), N = Q.size(2), d = Q.size(3);
	TORCH_CHECK(d % 16 == 0, "d must be a multiple of 16 for the tensor-core kernel, got ", d);

	// Tile sizes with the most resident warps per SM (ties: larger bc for fewer K/V steps, then
	// larger br); found once per d with the occupancy API and cached
	static int cached_d = -1, br = 0, bc = 0;
	if (d != cached_d){
		int M; // Bytes
		cudaDeviceGetAttribute(&M, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
		cudaFuncSetAttribute(flash_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, M); // Needed above 48 KB
		int best_warps = 0;
		br = bc = 0;
		for (int r : {64, 128}){
			for (int c : {16, 32, 48, 64}){
				int smem = sram_bytes(r, c, d);
				if (smem > M) continue;
				int blocks;
				cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, flash_attention, 2*r, smem);
				int warps = blocks*r/16;
				if (std::tie(warps, c, r) > std::tie(best_warps, bc, br)){
					best_warps = warps; bc = c; br = r;
				}
			}
		}
		TORCH_CHECK(best_warps > 0, "d = ", d, " is too large to fit in shared memory");
		cached_d = d;
	}
    float softmax_scale = 1.0f / sqrtf(d);

	torch::Tensor O = torch::empty_like(Q);

    const int tr = ceil((float)N / (float)br);
    const int tc = ceil((float)N / (float)bc);
    const int sram_size = sram_bytes(br, bc, d);

    dim3 grid_dim(tr, nh, B); // One block per Q tile; Q tiles of a head are consecutive
    dim3 block_dim(2*br);     // Two threads per row

    flash_attention<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), N, d, tc, bc, br, softmax_scale);
    return O;
}
