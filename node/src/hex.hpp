#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pouw {

using Hash = std::array<uint8_t, 32>;

std::string to_hex(std::span<const uint8_t> bytes);
std::vector<uint8_t> from_hex(std::string_view hex);
Hash hash_from_hex(std::string_view hex);  // throws unless exactly 64 hex characters

Hash sha256(std::span<const uint8_t> bytes);
Hash sha256(std::string_view text);

}  // namespace pouw
