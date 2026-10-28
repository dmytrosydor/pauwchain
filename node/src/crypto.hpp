#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "hex.hpp"

namespace pouw {

using PublicKey = Hash;  // Ed25519 public key; it is also the account address and the miner id
using Signature = std::array<uint8_t, 64>;

// Ed25519 (RFC 8032) key pair from a 32-byte seed. The seed is the only secret that is stored.
class KeyPair {
public:
  static KeyPair generate();  // seed from the OS random source
  static KeyPair from_seed(const Hash& seed);
  // Reads a key file (seed as 64 hex characters); creates it with mode 0600 if it does not exist.
  static KeyPair load_or_create(const std::string& path);
  static KeyPair load(const std::string& path);  // throws if the file is missing or malformed
  void save(const std::string& path) const;      // refuses to overwrite an existing file

  KeyPair(const KeyPair&) = default;
  KeyPair& operator=(const KeyPair&) = default;
  ~KeyPair();

  const PublicKey& public_key() const { return public_; }
  Signature sign(const Hash& message) const;

private:
  KeyPair() = default;
  Hash seed_{};
  PublicKey public_{};
};

bool verify(const PublicKey& key, const Hash& message, const Signature& sig);

Signature signature_from_hex(std::string_view hex);  // throws unless exactly 128 hex characters

}  // namespace pouw
