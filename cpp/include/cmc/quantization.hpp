#pragma once
#include "cmc/tensor.hpp"

namespace cmc{

struct QuantTensor{
    Tensor tensor;
    float scale;
};

float compute_scale(const Tensor& original);


QuantTensor quantize(const Tensor& original);
Tensor dequantize(const QuantTensor& quantized);

    
}

