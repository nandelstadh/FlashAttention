#pragma once

#include <torch/torch.h>
torch::Tensor forward_pass(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
