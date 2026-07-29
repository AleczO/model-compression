#pragma once
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>

namespace cmc{

struct RawTensor {
    std::string name;
    std::vector<uint32_t> shape;
    std::vector<float> data;
};

std::vector<RawTensor> load_weights(const std::string& path);

}


