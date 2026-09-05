#pragma once
#include "cmc/tensor.hpp"

namespace cmc{

float compute_scale(const Tensor& t_class);
Tensor quantize(const Tensor& t_class);
    
}

