cmake_minimum_required(VERSION 3.20)
project(pouwchain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_executable(node
  node/src/main.cpp
)

target_include_directories(node PRIVATE node/include)
