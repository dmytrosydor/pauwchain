#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "block.hpp"

namespace pouw {

// Tree of accepted blocks with the balances after each of them. The best chain is the longest; on equal
// length, the one with the smaller tip hash. Blocks arrive here already verified. Not thread-safe.
class Chain {
public:
  explicit Chain(Block genesis);

  bool contains(const Hash& h) const { return blocks_.count(h) != 0; }
  const Block* get(const Hash& h) const;
  const Block& tip() const { return blocks_.at(tip_); }
  const Block& genesis() const { return blocks_.at(genesis_); }
  uint64_t height() const { return tip().header.height; }

  // Adds a block whose parent is already present. Returns true if the tip changed. Throws if the parent is
  // unknown or the block's transfers do not apply to the parent's balances.
  bool add(const Block& b);

  const Ledger& ledger(const Hash& h) const { return ledgers_.at(h); }  // balances after block h
  const Ledger& tip_ledger() const { return ledgers_.at(tip_); }
  // The parent's balances with b applied, or the reason b's transfers are invalid.
  std::optional<std::string> try_apply(const Block& b, Ledger& out) const;

  std::vector<Block> best_chain() const;  // genesis..tip
  // Blocks between `old_tip` and the fork point with the current best chain: what a reorg abandoned.
  std::vector<Block> abandoned(const Hash& old_tip) const;
  std::vector<Block> best_from(uint64_t height) const;

  void save(const std::string& path) const;
  // Restores blocks from a file. Returns the number added (0 if there is no file); throws if the file is
  // unreadable or starts from another genesis, so the caller can keep it instead of overwriting it.
  size_t load(const std::string& path);

private:
  std::map<Hash, Block> blocks_;
  std::map<Hash, Ledger> ledgers_;
  Hash genesis_;
  Hash tip_;
};

}  // namespace pouw
