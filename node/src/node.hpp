#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "api.hpp"
#include "block.hpp"
#include "chain.hpp"
#include "events.hpp"
#include "p2p.hpp"
#include "worker_bridge.hpp"

namespace pouw {

struct Config {
  std::string id;
  uint16_t port = 0;
  std::vector<std::pair<std::string, uint16_t>> peers;
  std::string store_dir = "store";
  std::string task_path;
  std::string python = ".venv/bin/python";
  std::string worker_script = "worker/worker.py";
  std::string run_dir;            // task/result файли воркера, ланцюг і статус вузла
  std::string device = "cpu";
  bool lazy = false;              // атака 1: точки — копії w_start, без навчання
  double lazy_delay_s = 5.0;      // щоб лінивий вузол не засипав мережу блоками
  int q = 2;                      // скільки переходів перевіряє механізм B
  double alpha = 0.05;            // відносний поріг B (Р6), відкалібровано в E1 (docs/results.md)
  bool verify_a = false;          // додатково механізм A
  double a_eps = 0.02;
  int a_subset = 2000;
  uint64_t max_height = 0;        // 0 — майнити без кінця
  uint16_t api_port = 0;          // 0 — без HTTP API
};

Config parse_args(int argc, char** argv);

class Node {
public:
  explicit Node(Config cfg);
  void run();  // блокує: цикл майнінгу в поточному потоці

  // Для HTTP API; потокобезпечні.
  json status();
  json chain_from(uint64_t from_height);
  std::optional<json> block_json(const std::string& hash_hex);
  json admin(const json& request);

private:
  struct Incoming {
    int peer;
    Block block;
  };

  void on_connect(int peer);
  void on_message(int peer, const json& msg);
  void validator_loop();
  void mining_loop();

  // Повна перевірка: структура + B (+ A). Повертає причину відхилення або nullopt.
  std::optional<std::string> validate(const Block& b, const Block& parent, std::string& detail,
                                      json& checks);
  json status_locked() const;  // під mu_
  json worker_task(const char* op, const Block& b) const;
  void persist();  // під mu_
  void log(const std::string& line) const;
  void request_full_sync(int peer);

  Config cfg_;
  Task task_;
  Hash miner_id_{};
  std::unique_ptr<Chain> chain_;
  std::unique_ptr<WorkerBridge> miner_bridge_;
  std::unique_ptr<WorkerBridge> verifier_bridge_;
  std::unique_ptr<P2P> p2p_;
  std::unique_ptr<EventBus> events_;
  std::unique_ptr<Api> api_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Incoming> inbox_;
  std::set<Hash> rejected_;
  std::map<Hash, Incoming> orphans_;  // батько ще невідомий
  std::map<int, std::chrono::steady_clock::time_point> last_sync_;
  bool validating_ = false;  // валідатор зараз перевіряє блок (під mu_)
  std::atomic<bool> tip_changed_{false};
  std::atomic<bool> lazy_mode_{false};  // перемикається через API: вузол стає лінивим атакувальником
  std::atomic<bool> paused_{false};
  std::map<int, std::string> peer_ids_;  // peer → node_id з hello (під mu_)
  std::mt19937_64 rng_{std::random_device{}()};  // приватний вибір переходів (Р5)
  uint64_t mined_ = 0, accepted_ = 0, rejected_count_ = 0, lost_races_ = 0;
};

}  // namespace pouw
