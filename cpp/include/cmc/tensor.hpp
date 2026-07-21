#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace cmc {
    

class Tensor {
public:
    Tensor(std::vector<uint32_t> shape);
    Tensor(std::vector<uint32_t> shape, std::vector<float> data);

    float& at(std::initializer_list<uint32_t> indices);

    const std::vector<uint32_t>& shape() const;
    std::vector<float>& data();
    const std::vector<float>& data() const;

private:
    std::vector<uint32_t> shape_;
    std::vector<float> data_;

    size_t flatten_index(std::initializer_list<uint32_t> indices) const;
};


}  