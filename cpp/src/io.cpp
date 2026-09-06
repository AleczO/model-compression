#include <fstream>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <array>
#include "cmc/io.hpp"


namespace cmc{

namespace {

void check_magic(std::ifstream &f) {
    std::array<char, 4> magic{};
    f.read(magic.data(), magic.size());

    constexpr std::array<char, 4> expected = {'R', 'E', 'S', '8'};

    if(magic != expected){
        throw std::runtime_error("Invaid file format");
    }
}
    
uint32_t read_u32(std::ifstream& f) {
    uint32_t value;
    f.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void write_u32(std::ofstream& f, uint32_t value) {
    f.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

}



std::vector<cmc::Tensor> load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    check_magic(f);

    uint32_t version = read_u32(f);
    if (version != 1) {
        throw std::runtime_error("Unsupported file version: " + std::to_string(version));
    }

    uint32_t num_tensors = read_u32(f);

    std::vector<cmc::Tensor> tensors;
    tensors.reserve(num_tensors);

    for (uint32_t i = 0; i < num_tensors; ++i) {
        
        std::string name;
        uint32_t name_len = read_u32(f);
        name.resize(name_len);
        f.read(&name[0], name_len);    
        
        
        std::vector<uint32_t> shape;
        uint32_t ndim = read_u32(f);
        shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            shape[d] = read_u32(f);
        }

        std::vector<float> data;
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;

        data.resize(total_elements);
        f.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(float));
        Tensor t(std::move(name), std::move(shape), std::move(data));

        tensors.push_back(std::move(t));
    }

    return tensors;
}

void write_weights(const std::string& path, std::vector<Tensor>& net){
    std::ofstream f(path, std::ios::binary);
    std::string arch = "RES8";

    f.write(&arch[0], arch.size());

    write_u32(f, 1);
    write_u32(f, net.size());

    
    for (uint32_t i = 0; i < net.size(); ++i) {
        std::string name = net[i].name(); 
        write_u32(f, static_cast<uint32_t>(name.size()));
        f.write(&name[0], name.size());    
        
        std::vector<uint32_t> shape = net[i].shape();
        write_u32(f, static_cast<uint32_t>(shape.size()));
        for (uint32_t d = 0; d < shape.size(); ++d) {
            write_u32(f, shape[d]);
        }

        std::vector<float> data = net[i].data();
        f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    }

    return;
}

}

