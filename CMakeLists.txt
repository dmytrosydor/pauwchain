cmake_minimum_required(VERSION 3.20)
project(pouwchain LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# Header-only picosha2.h; SOURCE_SUBDIR points to a nonexistent directory so its examples are not built.
FetchContent_Declare(
  picosha2
  URL https://github.com/okdshin/PicoSHA2/archive/refs/tags/v1.0.1.tar.gz
  URL_HASH SHA256=9983136544234e573fe07cc1a22fdf978ad7979043e7be2e9082f6d4991ff8a8
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR none)

# Node HTTP API: header-only httplib.h, without building the library.
FetchContent_Declare(
  httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.1.tar.gz
  URL_HASH SHA256=405abd8170f2a446fc8612ac635d0db5947c0d2e156e32603403a4496255ff00
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR none)

# Ed25519 signatures (blocks, transactions, wallets): Monocypher, two C files, CC0/BSD-2.
FetchContent_Declare(
  monocypher
  URL https://github.com/LoupVaillant/Monocypher/archive/refs/tags/4.0.2.tar.gz
  URL_HASH SHA256=bc1ca30b1b2654e4e7daf2492c0d204200e55137f23fda6b7142fd7d523bd6b4
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR none)

FetchContent_MakeAvailable(nlohmann_json picosha2 httplib monocypher)

add_library(monocypher STATIC ${monocypher_SOURCE_DIR}/src/monocypher.c
                              ${monocypher_SOURCE_DIR}/src/optional/monocypher-ed25519.c)
target_include_directories(monocypher SYSTEM PUBLIC ${monocypher_SOURCE_DIR}/src ${monocypher_SOURCE_DIR}/src/optional)
find_package(Threads REQUIRED)

add_library(pouw STATIC
  node/src/hex.cpp
  node/src/crypto.cpp
  node/src/ledger.cpp
  node/src/mempool.cpp
  node/src/block.cpp
  node/src/chain.cpp
  node/src/worker_bridge.cpp
  node/src/p2p.cpp
  node/src/node.cpp
  node/src/events.cpp
  node/src/api.cpp
)
target_include_directories(pouw PUBLIC node/src)
target_include_directories(pouw SYSTEM PUBLIC ${picosha2_SOURCE_DIR} ${httplib_SOURCE_DIR})
target_link_libraries(pouw PUBLIC nlohmann_json::nlohmann_json Threads::Threads monocypher)
target_compile_options(pouw PRIVATE -Wall -Wextra -Wpedantic)

add_executable(node node/src/main.cpp)
target_link_libraries(node PRIVATE pouw)
target_compile_options(node PRIVATE -Wall -Wextra -Wpedantic)

# Wallet command line: keys, balances, transfers through a node's API.
add_executable(pouw-cli node/src/cli.cpp)
set_target_properties(pouw-cli PROPERTIES OUTPUT_NAME pouw)
target_link_libraries(pouw-cli PRIVATE pouw)
target_compile_options(pouw-cli PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(node_tests node/tests/test_main.cpp)
target_link_libraries(node_tests PRIVATE pouw)
target_compile_options(node_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME node_tests COMMAND node_tests ${CMAKE_SOURCE_DIR}/docs/golden.json
         ${CMAKE_SOURCE_DIR}/docs/vectors.json)
