#pragma once

#include <map>
#include <string>
#include <vector>

#include "block.hpp"

namespace pouw {

// Дерево прийнятих блоків. Найкращий ланцюг — найдовший; за рівної довжини — з меншим хешем
// вершини. Блоки сюди потрапляють уже перевіреними. Не потокобезпечний.
class Chain {
public:
  explicit Chain(Block genesis);

  bool contains(const Hash& h) const { return blocks_.count(h) != 0; }
  const Block* get(const Hash& h) const;
  const Block& tip() const { return blocks_.at(tip_); }
  const Block& genesis() const { return blocks_.at(genesis_); }
  uint64_t height() const { return tip().header.height; }

  // Додає блок, чий батько вже є. Повертає true, якщо змінилась вершина.
  bool add(const Block& b);

  std::vector<Block> best_chain() const;  // генезис..вершина
  std::vector<Block> best_from(uint64_t height) const;

  void save(const std::string& path) const;
  // Відновлює блоки з файлу, якщо генезис збігається. Повертає кількість доданих.
  size_t load(const std::string& path);

private:
  std::map<Hash, Block> blocks_;
  Hash genesis_;
  Hash tip_;
};

}  // namespace pouw
