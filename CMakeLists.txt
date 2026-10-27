cmake_minimum_required(VERSION 3.20)
project(pouwchain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# Лише заголовок picosha2.h; SOURCE_SUBDIR вказує на неіснуючу теку, щоб не збирати їхні приклади.
FetchContent_Declare(
  picosha2
  URL https://github.com/okdshin/PicoSHA2/archive/refs/tags/v1.0.1.tar.gz
  URL_HASH SHA256=9983136544234e573fe07cc1a22fdf978ad7979043e7be2e9082f6d4991ff8a8
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR none)

FetchContent_MakeAvailable(nlohmann_json picosha2)

add_library(pouw STATIC
  node/src/hex.cpp
  node/src/block.cpp
  node/src/chain.cpp
  node/src/worker_bridge.cpp
  node/src/p2p.cpp
  node/src/node.cpp
)
target_include_directories(pouw PUBLIC node/src)
target_include_directories(pouw SYSTEM PUBLIC ${picosha2_SOURCE_DIR})
target_link_libraries(pouw PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(pouw PRIVATE -Wall -Wextra -Wpedantic)

add_executable(node node/src/main.cpp)
target_link_libraries(node PRIVATE pouw)
target_compile_options(node PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(node_tests node/tests/test_main.cpp)
target_link_libraries(node_tests PRIVATE pouw)
target_compile_options(node_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME node_tests COMMAND node_tests ${CMAKE_SOURCE_DIR}/docs/golden.json
         ${CMAKE_SOURCE_DIR}/docs/vectors.json)
