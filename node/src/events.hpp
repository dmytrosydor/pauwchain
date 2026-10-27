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

// Node event log: each event gets a sequence number, is written to events.jsonl and
// kept in memory (the last kKeep) for API readers that lag behind or connect later.
class EventBus {
public:
  static constexpr size_t kKeep = 2000;

  explicit EventBus(const std::string& jsonl_path);

  // Adds seq and ts (unix, milliseconds) and wakes readers.
  void publish(const std::string& type, json data);

  // Events with seq > after; waits up to timeout_ms if there are none. Returns an empty vector on timeout.
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
