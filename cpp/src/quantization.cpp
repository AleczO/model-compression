#include "cmc/quantization.hpp"
#include <cmath>
#include <algorithm>

namespace cmc{
    

namespace {
    constexpr int Q_MAX = 127;  // symmetric INT8 range: [-127, 127]
}

float compute_scale(const Tensor& original){
    float max_val = 0.0f;

    for(auto t_val: original.data()){
        max_val = std::max(max_val, std::abs(t_val));
    }
    
    return max_val / static_cast<float>(127);
}

QuantTensor quantize(const Tensor& original){
    float s = compute_scale(original);
    Tensor quantized = original;

    for(auto& it: quantized.data()){
        it = std::round(it / s);
    }

    return {std::move(quantized), s};
}
    
Tensor dequantize(const QuantTensor& quantized){
    Tensor reconstructed = quantized.tensor;

    for(auto& it: reconstructed.data()){
        it = it * quantized.scale;
    }

    return reconstructed;
}

} //