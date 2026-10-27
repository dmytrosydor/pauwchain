#pragma once

#include <map>
#include <string>
#include <vector>

#include "block.hpp"

namespace pouw {

// Tree of accepted blocks. The best chain is the longest; on equal length, the one with the smaller
// tip hash. Blocks arrive here already verified. Not thread-safe.
class Chain {
public:
  explicit Chain(Block genesis);

  bool contains(const Hash& h) const { return blocks_.count(h) != 0; }
  const Block* get(const Hash& h) const;
  const Block& tip() const { return blocks_.at(tip_); }
  const Block& genesis() const { return blocks_.at(genesis_); }
  uint64_t height() const { return tip().header.height; }

  // Adds a block whose parent is already present. Returns true if the tip changed.
  bool add(const Block& b);

  std::vector<Block> best_chain() const;  // genesis..tip
  std::vector<Block> best_from(uint64_t height) const;

  void save(const std::string& path) const;
  // Restores blocks from a file if the genesis matches. Returns the number added.
  size_t load(const std::string& path);

private:
  std::map<Hash, Block> blocks_;
  Hash genesis_;
  Hash tip_;
};

}  // namespace pouw
