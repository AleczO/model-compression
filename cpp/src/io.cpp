#include <fstream>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include "cmc/io.hpp"


namespace cmc{

namespace {
    
uint32_t read_u32(std::ifstream& f) {
    uint32_t value;
    f.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

}


std::vector<RawTensor> load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    uint32_t num_tensors = read_u32(f);
    std::vector<RawTensor> tensors;

    for (uint32_t i = 0; i < num_tensors; ++i) {
        RawTensor t;

        // uint32_t name_len = read_u32(f);
        // t.name.resize(name_len);
        // f.read(&t.name[0], name_len);

        uint32_t ndim = read_u32(f);
        t.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            t.shape[d] = read_u32(f);
        }

        uint32_t total_elements = 1;
        for (auto d : t.shape) total_elements *= d;

        t.data.resize(total_elements);
        f.read(reinterpret_cast<char*>(t.data.data()), total_elements * sizeof(float));

        tensors.push_back(std::move(t));
    }

    return tensors;
}

}

