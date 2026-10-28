#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "crypto.hpp"

namespace pouw {

using json = nlohmann::json;

// Coins are whole units. Every accepted block mints kBlockReward for its miner, plus the fees of its transactions.
inline constexpr uint64_t kBlockReward = 50;
inline constexpr size_t kMaxTxsPerBlock = 100;

// A signed transfer between accounts (docs/contract.md §5). Accounts are Ed25519 public keys.
struct Tx {
  Hash chain{};  // task_id of the network: a transfer signed for one network is invalid on another
  PublicKey from{};
  PublicKey to{};
  uint64_t amount = 0;
  uint64_t fee = 0;
  uint64_t nonce = 0;  // the sender's transfer counter: each value is spent once, in order
  Signature sig{};

  static constexpr size_t kSize = 4 + 32 * 3 + 8 * 3;  // "PWTX" + fields, little-endian
  std::array<uint8_t, kSize> serialize() const;      // what the signature covers (without sig)
  Hash id() const { return sha256(serialize()); }
  // Covers the signature too. tx_root is built from these, so the block hash fixes every byte of its
  // transfers: a relayed copy with a mangled signature is a different block, not the same one gone bad.
  Hash full_hash() const;
  void sign(const KeyPair& key);
  bool signature_ok() const { return verify(from, id(), sig); }
};

json tx_to_json(const Tx& tx);
Tx tx_from_json(const json& j);  // throws on malformed input
Hash tx_root_of(const std::vector<Tx>& txs);

struct Account {
  uint64_t balance = 0;
  uint64_t nonce = 0;  // nonce of the next transfer from this account
};

// Balances after some block. Copied per block; small, since a test network has few accounts.
class Ledger {
public:
  const Account& account(const PublicKey& key) const;
  const std::map<PublicKey, Account>& accounts() const { return accounts_; }
  uint64_t supply() const { return supply_; }

  // Stateless checks of a transfer: network, signature, no self-transfer, no overflow.
  static std::optional<std::string> check_tx(const Tx& tx, const Hash& chain);
  // Applies a block's transfers in order, then pays the miner reward + fees. On error the ledger is unchanged.
  std::optional<std::string> apply(const std::vector<Tx>& txs, const PublicKey& miner, const Hash& chain);

private:
  std::map<PublicKey, Account> accounts_;
  uint64_t supply_ = 0;
};

}  // namespace pouw
