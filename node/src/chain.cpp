#include "chain.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace pouw {

Chain::Chain(Block genesis) : genesis_(genesis.hash()), tip_(genesis_) {
  blocks_.emplace(genesis_, std::move(genesis));
  ledgers_.emplace(genesis_, Ledger{});
}

std::optional<std::string> Chain::try_apply(const Block& b, Ledger& out) const {
  auto it = ledgers_.find(b.header.prev_hash);
  if (it == ledgers_.end()) return "parent unknown";
  out = it->second;
  return out.apply(b.txs, b.header.miner_id, genesis().header.task_id);
}

const Block* Chain::get(const Hash& h) const {
  auto it = blocks_.find(h);
  return it == blocks_.end() ? nullptr : &it->second;
}

bool Chain::add(const Block& b) {
  Hash h = b.hash();
  if (contains(h)) return false;
  if (!contains(b.header.prev_hash)) throw std::invalid_argument("parent unknown");
  Ledger next;
  if (auto err = try_apply(b, next)) throw std::invalid_argument("ledger: " + *err);
  blocks_.emplace(h, b);
  ledgers_.emplace(h, std::move(next));
  // Longest chain; on equal height, the smaller block hash wins. The same rule on every node
  // lets them converge on one tip even when miners finish a round at the same time.
  if (b.header.height > height() || (b.header.height == height() && h < tip_)) {
    tip_ = h;
    return true;
  }
  return false;
}

std::vector<Block> Chain::best_chain() const {
  std::vector<Block> out;
  for (const Block* b = &tip(); b; b = b->header.height == 0 ? nullptr : get(b->header.prev_hash)) {
    out.push_back(*b);
  }
  std::reverse(out.begin(), out.end());
  return out;
}

std::vector<Block> Chain::abandoned(const Hash& old_tip) const {
  std::set<Hash> best;
  for (const auto& b : best_chain()) best.insert(b.hash());
  std::vector<Block> out;
  for (const Block* b = get(old_tip); b && !best.count(b->hash()); b = get(b->header.prev_hash)) out.push_back(*b);
  return out;
}

std::vector<Block> Chain::best_from(uint64_t height) const {
  auto all = best_chain();
  if (height >= all.size()) return {};
  return {all.begin() + static_cast<std::ptrdiff_t>(height), all.end()};
}

void Chain::save(const std::string& path) const {
  json blocks = json::array();
  for (const auto& b : best_chain()) blocks.push_back(block_to_json(b));
  json j = {{"height", height()}, {"tip", to_hex(tip_)}, {"blocks", blocks}};
  auto dir = std::filesystem::path(path).parent_path();
  if (!dir.empty()) std::filesystem::create_directories(dir);
  std::string tmp = path + ".tmp";
  std::ofstream(tmp) << j.dump(2) << '\n';
  std::filesystem::rename(tmp, path);
}

size_t Chain::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) return 0;
  json j = json::parse(in, nullptr, false);
  if (j.is_discarded() || !j.contains("blocks") || j.at("blocks").empty()) {
    throw std::runtime_error(path + " is not a saved chain");
  }
  const auto& blocks = j.at("blocks");
  if (block_from_json(blocks.front()).hash() != genesis_) {
    throw std::runtime_error(path + " belongs to another network or an older protocol version");
  }
  size_t added = 0;
  for (size_t i = 1; i < blocks.size(); ++i) {
    try {
      Block b = block_from_json(blocks[i]);
      if (!contains(b.header.prev_hash)) break;
      add(b);
    } catch (const std::exception&) {
      break;  // a saved block that no longer parses or applies (edited file): keep the valid prefix
    }
    ++added;
  }
  return added;
}

}  // namespace pouw
