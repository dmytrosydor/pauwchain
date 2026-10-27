#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "block.hpp"

namespace pouw {

// Журнал подій вузла: кожна подія отримує порядковий номер, пишеться в events.jsonl і
// лишається в пам'яті (останні kKeep) для читачів API, які відстають або підключаються пізніше.
class EventBus {
public:
  static constexpr size_t kKeep = 2000;

  explicit EventBus(const std::string& jsonl_path);

  // Додає seq і ts (unix, мілісекунди) і будить читачів.
  void publish(const std::string& type, json data);

  // Події з seq > after; чекає до timeout_ms, якщо нових немає. Повертає порожній вектор за тайм-аутом.
  std::vector<json> wait_after(uint64_t after, int timeout_ms);
  uint64_t last_seq() const;

private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<json> recent_;
  uint64_t seq_ = 0;
  std::ofstream out_;
};

}  // namespace pouw
