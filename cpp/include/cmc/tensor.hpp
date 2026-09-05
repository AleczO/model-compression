#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

namespace cmc {
    

class Tensor {
public:
    Tensor(std::string name, std::vector<uint32_t> shape);
    Tensor(std::string name, std::vector<uint32_t> shape, std::vector<float> data);

    float& at(std::initializer_list<uint32_t> indices);

    const std::vector<uint32_t>& shape() const;
    std::vector<float>& data();
    const std::vector<float>& data() const;
    const std::string& name();

private:

    std::string name_;
    std::vector<uint32_t> shape_;
    std::vector<float> data_;
    
    size_t flatten_index(std::initializer_list<uint32_t> indices) const;
};


}  