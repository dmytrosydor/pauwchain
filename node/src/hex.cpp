#include "hex.hpp"

#include <picosha2.h>

#include <stdexcept>

namespace pouw {

std::string to_hex(std::span<const uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (uint8_t b : bytes) {
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 0xf]);
  }
  return out;
}

static int nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  throw std::invalid_argument("invalid hex digit");
}

std::vector<uint8_t> from_hex(std::string_view hex) {
  if (hex.size() % 2 != 0) throw std::invalid_argument("odd hex length");
  std::vector<uint8_t> out(hex.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<uint8_t>(nibble(hex[2 * i]) << 4 | nibble(hex[2 * i + 1]));
  }
  return out;
}

Hash hash_from_hex(std::string_view hex) {
  if (hex.size() != 64) throw std::invalid_argument("hash must be 64 hex chars");
  auto bytes = from_hex(hex);
  Hash h;
  std::copy(bytes.begin(), bytes.end(), h.begin());
  return h;
}

Hash sha256(std::span<const uint8_t> bytes) {
  Hash h;
  picosha2::hash256(bytes.begin(), bytes.end(), h.begin(), h.end());
  return h;
}

Hash sha256(std::string_view text) {
  return sha256(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

}  // namespace pouw
