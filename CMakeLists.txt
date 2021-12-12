cmake_minimum_required(VERSION 3.2)
project(libGaussianProcess VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(GPModel
    src/GaussianProcessModel.cpp
    include/GaussianProcess/GaussianProcessModel.hpp
)

add_library(GPModel::GPModel ALIAS GPModel)

# Define headers for this library.
target_include_directories(GPModel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# set_target_properties(GPModel PROPERTIES
#     CXX_STANDARD 20
# )

add_subdirectory(test)

