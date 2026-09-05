#include "cmc/metrics.hpp"
#include <stdexcept>

namespace cmc{

float mse(const Tensor& original, const Tensor& reconstructed){
    if (original.data().size() != reconstructed.data().size())
        throw std::invalid_argument("mse: tensors have different sizes");

    const auto& orig_data = original.data();
    const auto& recon_data = reconstructed.data();

    size_t n = original.data().size();

    float sum = 0.0f;
    for(size_t  i = 0; i < n; i++){
        float diff = orig_data[i] - recon_data[i];
        sum += diff * diff;
    }

    return sum / static_cast<float>(n);
}

}