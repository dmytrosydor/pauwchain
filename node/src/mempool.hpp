#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ledger.hpp"

namespace pouw {

// Transfers waiting for a block (docs/contract.md §5). Two lists per sender, both ordered by nonce:
//   ready   — continues the sender's nonce at the tip without gaps and is funded: a miner can take it as is;
//   waiting — has a gap before it or no funds yet (e.g. the coins are in a block this node is still checking).
// Only ready transfers are relayed and mined. Not thread-safe.
class Mempool {
public:
  static constexpr size_t kMaxReady = 5000;
  static constexpr size_t kMaxWaiting = 1000;
  static constexpr size_t kMaxWaitingPerSender = 16;
  static constexpr uint64_t kMaxNonceAhead = 64;

  enum class Result { ready, waiting, rejected };
  struct Outcome {
    Result result;
    std::string reason;  // why it waits or was rejected
  };

  // A new transfer from a wallet or a peer; the signature is checked here.
  Outcome offer(const Tx& tx, const Ledger& tip, const Hash& chain);
  // After the tip changed (a new block or a reorg): rebuilds both lists against the new balances. `returned`
  // are transfers from blocks that left the best chain; they get another chance. Returns the transfers that
  // became ready now, so the node can relay them.
  std::vector<Tx> retarget(const Ledger& tip, const std::vector<Tx>& returned);
  // Transfers for a block on `parent` (the current tip), at most kMaxTxsPerBlock.
  std::vector<Tx> select(const Ledger& parent) const;
  uint64_t next_nonce(const PublicKey& key, const Ledger& tip) const;  // for a wallet's next transfer

  size_t ready_count() const { return ready_count_; }
  size_t waiting_count() const { return waiting_count_; }
  std::vector<Tx> ready() const;
  std::vector<Tx> waiting() const;

private:
  using Queue = std::map<uint64_t, Tx>;  // nonce → transfer
  bool add_waiting(const Tx& tx);
  // Moves waiting transfers of `from` that now continue its ready list; appends them to `promoted`.
  void promote(const PublicKey& from, const Ledger& tip, std::vector<Tx>* promoted);
  uint64_t spent(const PublicKey& from) const;  // amount + fee of the sender's ready transfers

  std::map<PublicKey, Queue> ready_, waiting_;
  size_t ready_count_ = 0, waiting_count_ = 0;
};

}  // namespace pouw
