#include "cmc/tensor.hpp"
#include <stdexcept>
#include <numeric>

namespace cmc {


Tensor::Tensor(std::string name, std::vector<uint32_t> shape)
    : name_(std::move(name)), shape_(std::move(shape)) {
    size_t total = std::accumulate(shape_.begin(), shape_.end(),
                                    size_t(1), std::multiplies<>());
    data_.resize(total);
}

Tensor::Tensor(std::string name, std::vector<uint32_t> shape, std::vector<float> data) 
:  name_(std::move(name)), shape_(std::move(shape)), data_(std::move(data)) {
     size_t expected = std::accumulate(shape_.begin(), shape_.end(),
                                        size_t(1), std::multiplies<>());
    if (data_.size() != expected)
        throw std::invalid_argument("Tensor: data size does not match shape");
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

const std::string& Tensor::name() {
    return name_;
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