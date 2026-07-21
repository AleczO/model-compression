#include "cmc/tensor.hpp"
#include <numeric>

namespace cmc {

Tensor::Tensor(std::vector<uint32_t> shape)
    : shape_(std::move(shape)) {
}

Tensor::Tensor(std::vector<uint32_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
}

float& Tensor::at(std::initializer_list<uint32_t> indices) {
    return data_[flatten_index(indices)];
}

const std::vector<uint32_t>& Tensor::shape() const {
    return shape_;
}

std::vector<float>& Tensor::data() {
    return data_;
}

const std::vector<float>& Tensor::data() const {
    return data_;
}

size_t Tensor::flatten_index(std::initializer_list<uint32_t> indices) const {
    size_t idx = 0, stride = 1;
    auto it = indices.end();

    for (auto s = shape_.rbegin(); s != shape_.rend(); ++s) {
        --it;
        idx += (*it) * stride;
        stride *= *s;
    }
    
    return idx;
}

}  // namespace cmc