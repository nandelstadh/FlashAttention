#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "flash.cuh"

enum class Mode { CPU, Unfused, FA, Test };

[[noreturn]] static void usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s <mode> B nh N d\n"
                 "  cpu      benchmark unfused PyTorch attention on the CPU\n"
                 "  unfused  benchmark unfused PyTorch attention on the GPU\n"
                 "  fa       benchmark the FlashAttention kernel\n"
                 "  test     check the FlashAttention kernel against unfused GPU attention\n",
                 prog);
    std::exit(EXIT_FAILURE);
}

static Mode parse_mode(const char* s, const char* prog) {
    if (std::strcmp(s, "cpu") == 0) return Mode::CPU;
    if (std::strcmp(s, "unfused") == 0) return Mode::Unfused;
    if (std::strcmp(s, "fa") == 0) return Mode::FA;
    if (std::strcmp(s, "test") == 0) return Mode::Test;
    std::fprintf(stderr, "Invalid mode: '%s'\n", s);
    usage(prog);
}

// Parsing the tensor dimensions
static int parse_positive_int(const char* s, const char* name) {
    char* end;
    long v = std::strtol(s, &end, 10);
    if (*end != '\0' || v <= 0) {
        std::fprintf(stderr, "Invalid value for %s: '%s' (expected a positive integer)\n", name, s);
        std::exit(EXIT_FAILURE);
    }
    return (int)v;
}

torch::Tensor unfused_attn(const torch::Tensor Q,
                          const torch::Tensor K,
                          const torch::Tensor V) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(K.size(-1)));
    auto att = torch::matmul(Q, K.transpose(-2, -1)) * scale;
    att = torch::softmax(att, /*dim=*/-1);
    return torch::matmul(att, V);
}

constexpr double TIME_BUDGET_MS = 1000.0; // Target total timed duration
constexpr int MAX_ITERS = 100;

// Returns average ms per call; iters is set to the number of timed calls
template <typename F>
static double time_ms(F&& fn, bool cuda, int& iters) {
    auto run = [&](int n) {
        if (cuda) cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < n; it++) fn();
        if (cuda) cudaDeviceSynchronize(); // GPU launches are async
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    run(1);                 // Warm-up (cuBLAS init, allocator, caches)
    double single = run(1); // Calibration
    iters = (int)std::clamp(TIME_BUDGET_MS / single, 1.0, (double)MAX_ITERS);
    if (iters == 1) return single; // Slow run: the calibration run is the measurement
    return run(iters) / iters;
}

int main(int argc, char** argv) {
    if (argc != 6) usage(argv[0]);
    const Mode mode = parse_mode(argv[1], argv[0]);
    const int B = parse_positive_int(argv[2], "B");
    const int nh = parse_positive_int(argv[3], "nh");
    const int N = parse_positive_int(argv[4], "N");
	const int d = parse_positive_int(argv[5], "d");

	const bool cuda = mode != Mode::CPU;
	torch::Device device(cuda ? torch::kCUDA : torch::kCPU);
	torch::Tensor Q = torch::randn({B, nh, N, d}, device);
	torch::Tensor K = torch::randn({B, nh, N, d}, device);
	torch::Tensor V = torch::randn({B, nh, N, d}, device);

	if (mode == Mode::Test){
		auto manual_result = unfused_attn(Q, K, V);
		auto fa_result = forward_pass(Q, K, V);
		// TF32 tensor cores keep a 10-bit mantissa, so allow ~1e-3 level error
		bool ok = torch::allclose(fa_result, manual_result, 5e-3, 5e-3);
		std::printf("max abs diff: %g -> %s\n",
					(fa_result - manual_result).abs().max().item<float>(), ok ? "PASS" : "FAIL");
		return ok ? EXIT_SUCCESS : EXIT_FAILURE;
	}

	int iters;
	double ms = (mode == Mode::FA)
		? time_ms([&]{ forward_pass(Q, K, V); }, cuda, iters)
		: time_ms([&]{ unfused_attn(Q, K, V); }, cuda, iters);

	const double flops = 4.0 * B * nh * (double)N * N * d; // The two matmuls
	std::printf("%s: %.3f ms/iter over %d iters, %.1f GFLOP/s\n",
				argv[1], ms, iters, flops / (ms * 1e6));
	return EXIT_SUCCESS;
}
