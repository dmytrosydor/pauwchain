#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "hex.hpp"
#include "ledger.hpp"

namespace pouw {

// Version 1 (the thesis protocol) is 236 bytes. Version 2 adds tx_root: the block also carries transfers.
inline constexpr size_t kHeaderSizeV1 = 236;
inline constexpr size_t kHeaderSizeV2 = kHeaderSizeV1 + 32;
inline constexpr uint32_t kBlockVersion = 2;

// Block header, docs/contract.md §2. Fields in serialization order.
struct Header {
  uint32_t version = kBlockVersion;
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
  Hash tx_root{};  // version ≥ 2 only

  std::vector<uint8_t> serialize() const;  // canonical little-endian form
  Hash hash() const;
};

struct Block {
  Header header;
  std::vector<Hash> ckpt_hashes;  // C₀..C_m, stored in store/<hex>.safetensors
  std::vector<Tx> txs;            // committed by header.tx_root
  Signature sig{};                // the miner's signature of the block hash (miner_id is its public key)

  Hash hash() const { return header.hash(); }
  void sign(const KeyPair& key) { sig = key.sign(hash()); }
};

// Training task parameters; identical on all nodes of the network.
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

// Genesis is identical on all nodes: height 0, weights C₀ = genesis state from task_seed.
Block make_genesis(const Task& task, const Hash& w0);

// Structural rules of a block relative to its parent, including the miner's signature and tx_root.
// Balances are checked separately, against the parent's ledger (Chain::ledger). Returns a violation description or nullopt.
std::optional<std::string> check_structure(const Block& b, const Block& parent, const Task& task);

}  // namespace pouw
