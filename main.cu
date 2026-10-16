#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <torch/torch.h>
#include "flash.h"

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

torch::Tensor manual_attn(const torch::Tensor Q,
                          const torch::Tensor K,
                          const torch::Tensor V) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(K.size(-1)));
    auto att = torch::matmul(Q, K.transpose(-2, -1)) * scale;
    att = torch::softmax(att, /*dim=*/-1);
    return torch::matmul(att, V);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: %s B nh N d\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int B = parse_positive_int(argv[1], "B");
    const int nh = parse_positive_int(argv[2], "nh");
    const int N = parse_positive_int(argv[3], "N");
	const int d = parse_positive_int(argv[4], "d");

	torch::Device device(torch::kCUDA);
	torch::Tensor Q = torch::randn({B, nh, N, d}, device);
	torch::Tensor K = torch::randn({B, nh, N, d}, device);
	torch::Tensor V = torch::randn({B, nh, N, d}, device);

	auto manual_result = manual_attn(Q, K, V);
	auto fa_result = forward_pass(Q, K, V);
	return torch::allclose(fa_result, manual_result, 1e-3, 1e-3);
}
