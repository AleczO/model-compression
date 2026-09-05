#include "cmc/quantization.hpp"
#include <cmath>
#include <algorithm>

namespace cmc{
    

namespace {
    constexpr int Q_MAX = 127;  // symmetric INT8 range: [-127, 127]
}

float compute_scale(const Tensor& t_class){
    float max_val = 0.0f;

    for(auto t_val: t_class.data()){
        max_val = std::max(max_val, std::abs(t_val));
    }
    
    return max_val / static_cast<float>(127);
}

Tensor quantize(const Tensor& t_class){
    Tensor quant_t_class = t_class;

    for(auto& it: quant_t_class.data()){
        it = std::round(it / compute_scale(t_class));
    }

    return quant_t_class;
}
    
} //