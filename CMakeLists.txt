cmake_minimum_required(VERSION 3.13)

project("Regression")
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(regression src/regression.cpp src/readfile.cpp)

target_link_libraries(regression Eigen3::Eigen)
