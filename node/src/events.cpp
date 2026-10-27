#include "events.hpp"

#include <chrono>

namespace pouw {

EventBus::EventBus(const std::string& jsonl_path) : out_(jsonl_path, std::ios::app) {}

void EventBus::publish(const std::string& type, json data) {
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
  {
    std::lock_guard lock(mu_);
    json ev = {{"seq", ++seq_}, {"ts", ms}, {"type", type}, {"data", std::move(data)}};
    if (out_) out_ << ev.dump() << '\n' << std::flush;
    recent_.push_back(std::move(ev));
    if (recent_.size() > kKeep) recent_.pop_front();
  }
  cv_.notify_all();
}

std::vector<json> EventBus::wait_after(uint64_t after, int timeout_ms) {
  std::unique_lock lock(mu_);
  cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return seq_ > after; });
  std::vector<json> out;
  for (const auto& ev : recent_) {
    if (ev.at("seq").get<uint64_t>() > after) out.push_back(ev);
  }
  return out;
}

uint64_t EventBus::last_seq() const {
  std::lock_guard lock(mu_);
  return seq_;
}

}  // namespace pouw
