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

static constexpr char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string to_base64(std::span<const uint8_t> bytes) {
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < bytes.size(); i += 3) {
    uint32_t v = bytes[i] << 16 | bytes[i + 1] << 8 | bytes[i + 2];
    for (int s = 18; s >= 0; s -= 6) out.push_back(kB64[v >> s & 63]);
  }
  if (size_t rest = bytes.size() - i) {
    uint32_t v = bytes[i] << 16 | (rest == 2 ? bytes[i + 1] << 8 : 0);
    out.push_back(kB64[v >> 18 & 63]);
    out.push_back(kB64[v >> 12 & 63]);
    out.push_back(rest == 2 ? kB64[v >> 6 & 63] : '=');
    out.push_back('=');
  }
  return out;
}

static int sextet(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  throw std::invalid_argument("invalid base64 character");
}

std::vector<uint8_t> from_base64(std::string_view text) {
  if (text.size() % 4 != 0) throw std::invalid_argument("base64 length must be a multiple of 4");
  size_t pad = text.empty() ? 0 : (text.back() == '=') + (text.size() > 1 && text[text.size() - 2] == '=');
  std::vector<uint8_t> out;
  out.reserve(text.size() / 4 * 3);
  for (size_t i = 0; i < text.size(); i += 4) {
    const bool last = i + 4 == text.size();
    uint32_t v = 0;
    for (size_t j = 0; j < 4; ++j) {
      char c = text[i + j];
      v = v << 6 | (c == '=' && last && j >= 4 - pad ? 0 : sextet(c));
    }
    out.push_back(static_cast<uint8_t>(v >> 16));
    if (!last || pad < 2) out.push_back(static_cast<uint8_t>(v >> 8));
    if (!last || pad < 1) out.push_back(static_cast<uint8_t>(v));
  }
  return out;
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
