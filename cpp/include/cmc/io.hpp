#pragma once
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>
#include "cmc/tensor.hpp"

namespace cmc{

std::vector<cmc::Tensor> load_weights(const std::string& path);

void write_weights(const std::string& path, std::vector<Tensor>& net);

}


