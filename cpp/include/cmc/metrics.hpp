#pragma once
#include "cmc/tensor.hpp"


namespace cmc{

float mse(const Tensor& original, const Tensor& reconstructed);

}