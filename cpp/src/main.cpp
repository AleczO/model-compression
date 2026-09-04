#include <iostream>
#include "cmc/io.hpp"
#include "cmc/tensor.hpp"

int main(){

    std::vector<cmc::RawTensor> weights = cmc::load_weights("../../data/exported/weights.bin");

    for(auto i: weights){
        for(auto j: i.shape)
            std::cout << j << ' ';
        std::cout << '\n';
    }


    std::cout << "Hello, World!" << std::endl;
    return 0;
}