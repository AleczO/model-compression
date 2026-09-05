cmake_minimum_required(VERSION 3.20)
project(model_compression CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

add_library(cmc_core
    src/tensor.cpp
    src/io.cpp
    src/quantization.cpp
)

target_include_directories(cmc_core PUBLIC include)

add_executable(cmc_main src/main.cpp)
target_link_libraries(cmc_main PRIVATE cmc_core)