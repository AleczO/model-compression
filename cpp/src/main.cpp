#include <iostream>
#include "cmc/io.hpp"
#include "cmc/tensor.hpp"
#include "cmc/quantization.hpp"

int main(){

    std::vector<cmc::Tensor> Res8 = cmc::load_weights("../../data/exported/weights.bin");

    for(auto& it: Res8){
        std::cout << it.name() << std::endl;
    }

    cmc::Tensor Layer = Res8[0];
    cmc::Tensor LayerQuant = cmc::quantize(Layer);

    for(auto& it: LayerQuant.data())
        std::cout << it << '\n';
    
    return 0;
}