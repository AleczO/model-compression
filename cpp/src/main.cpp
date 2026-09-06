#include <iostream>
#include "cmc/io.hpp"
#include "cmc/tensor.hpp"
#include "cmc/quantization.hpp"
#include "cmc/metrics.hpp"

int main(){
    std::vector<cmc::Tensor> Res8 = cmc::load_weights("../../data/exported/weights.bin");
    std::vector<cmc::Tensor> DequantRes8(Res8);
    
    for(auto &layer: DequantRes8){
        cmc::QuantTensor quantized = cmc::quantize(layer);
        cmc::Tensor reconstructed = cmc::dequantize(quantized);
        layer = reconstructed;
    }

    for (size_t i = 0; i < Res8.size(); ++i) {
        std::cout << Res8[i].name() << ": " << cmc::mse(Res8[i], DequantRes8[i]) << '\n';
    }

    cmc::write_weights("../../data/exported/reconstructed_weights.bin", DequantRes8);
    
    return 0;
}