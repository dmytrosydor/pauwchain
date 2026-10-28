#include "crypto.hpp"

#include <monocypher-ed25519.h>
#include <monocypher.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace pouw {

KeyPair KeyPair::generate() {
  Hash seed;
  std::ifstream rnd("/dev/urandom", std::ios::binary);
  if (!rnd.read(reinterpret_cast<char*>(seed.data()), seed.size())) {
    throw std::runtime_error("cannot read /dev/urandom");
  }
  KeyPair k = from_seed(seed);
  crypto_wipe(seed.data(), seed.size());
  return k;
}

KeyPair::~KeyPair() { crypto_wipe(seed_.data(), seed_.size()); }

KeyPair KeyPair::from_seed(const Hash& seed) {
  KeyPair k;
  k.seed_ = seed;
  Hash tmp = seed;  // Monocypher wipes the seed buffer it is given
  std::array<uint8_t, 64> secret;
  crypto_ed25519_key_pair(secret.data(), k.public_.data(), tmp.data());
  crypto_wipe(secret.data(), secret.size());
  return k;
}

KeyPair KeyPair::load(const std::string& path) {
  std::ifstream in(path);
  std::string hex;
  if (!(in >> hex)) throw std::runtime_error("cannot read key file " + path);
  try {
    return from_seed(hash_from_hex(hex));
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("key file " + path + " is malformed (expected 64 hex characters)");
  }
}

KeyPair KeyPair::load_or_create(const std::string& path) {
  if (std::filesystem::exists(path)) return load(path);
  KeyPair k = generate();
  k.save(path);
  return k;
}

void KeyPair::save(const std::string& path) const {
  auto dir = std::filesystem::path(path).parent_path();
  if (!dir.empty()) std::filesystem::create_directories(dir);
  // O_EXCL: never overwrite a key; mode 0600 from the start, so the seed is never readable by others.
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) {
    throw std::runtime_error(errno == EEXIST ? "refusing to overwrite " + path
                                             : "cannot write key file " + path + ": " + std::strerror(errno));
  }
  std::string line = to_hex(seed_) + "\n";
  bool ok = ::write(fd, line.data(), line.size()) == static_cast<ssize_t>(line.size());
  ok = ::close(fd) == 0 && ok;
  crypto_wipe(line.data(), line.size());
  if (!ok) throw std::runtime_error("cannot write key file " + path);
}

Signature KeyPair::sign(const Hash& message) const {
  Hash tmp = seed_;
  std::array<uint8_t, 64> secret;
  PublicKey pub;
  crypto_ed25519_key_pair(secret.data(), pub.data(), tmp.data());
  Signature sig;
  crypto_ed25519_sign(sig.data(), secret.data(), message.data(), message.size());
  crypto_wipe(secret.data(), secret.size());
  return sig;
}

bool verify(const PublicKey& key, const Hash& message, const Signature& sig) {
  return crypto_ed25519_check(sig.data(), key.data(), message.data(), message.size()) == 0;
}

Signature signature_from_hex(std::string_view hex) {
  if (hex.size() != 128) throw std::invalid_argument("signature must be 128 hex chars");
  auto bytes = from_hex(hex);
  Signature s;
  std::copy(bytes.begin(), bytes.end(), s.begin());
  return s;
}

}  // namespace pouw
