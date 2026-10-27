#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "hex.hpp"

namespace pouw {

using json = nlohmann::json;

inline constexpr size_t kHeaderSize = 236;

// Заголовок блоку, docs/contract.md §2. Поля в порядку серіалізації.
struct Header {
  uint32_t version = 1;
  uint64_t height = 0;
  Hash prev_hash{};
  Hash task_id{};
  uint64_t round = 0;
  Hash miner_id{};
  uint64_t timestamp = 0;
  Hash w_start{};
  Hash w_end{};
  Hash ckpt_root{};
  uint32_t K = 0;
  uint32_t k = 0;
  int64_t metric_micro = 0;

  std::array<uint8_t, kHeaderSize> serialize() const;  // канонічна little-endian форма
  Hash hash() const;
};

struct Block {
  Header header;
  std::vector<Hash> ckpt_hashes;  // C₀..C_m, лежать у store/<hex>.safetensors

  Hash hash() const { return header.hash(); }
};

// Параметри завдання навчання; однакові в усіх вузлів мережі.
struct Task {
  Hash task_id{};
  uint64_t task_seed = 0;
  uint32_t K = 0;
  uint32_t k = 0;
  uint32_t batch_size = 64;
  double lr = 0.05;
  std::string dataset = "mnist";
};

Task task_from_json(const json& j, const std::string& raw_text);

json header_to_json(const Header& h);
Header header_from_json(const json& j);
json block_to_json(const Block& b);
Block block_from_json(const json& j);

Hash ckpt_root_of(const std::vector<Hash>& ckpt_hashes);

// Генезис однаковий на всіх вузлах: висота 0, ваги C₀ = генезис-стан з task_seed.
Block make_genesis(const Task& task, const Hash& w0);

// Структурні правила блоку відносно батька. Повертає опис порушення або nullopt.
std::optional<std::string> check_structure(const Block& b, const Block& parent, const Task& task);

}  // namespace pouw
