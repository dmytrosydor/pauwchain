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
  std::string listen = "127.0.0.1";      // P2P address; 0.0.0.0 to accept peers from other machines
  std::string api_listen = "127.0.0.1";  // the API has an unauthenticated /admin, so it stays local by default
  std::vector<std::pair<std::string, uint16_t>> peers;
  std::string store_dir = "store";
  std::string task_path;
  std::string python = ".venv/bin/python";
  std::string worker_script = "worker/worker.py";
  std::string worker_server = "worker/server.py";
  std::string worker_mode = "server";  // server: persistent worker on a Unix socket; spawn: a process per call
  std::string run_dir;            // worker task/result files, chain and node status
  std::string device = "cpu";
  bool lazy = false;              // attack 1: checkpoints are copies of w_start, no training
  double lazy_delay_s = 1.0;      // keeps a lazy node from flooding the network with blocks
  int q = 2;                      // how many transitions mechanism B checks
  double alpha = 0.05;            // relative threshold of B (D6), calibrated in E1 (docs/results.md)
  bool verify_a = false;          // also run mechanism A
  double a_eps = 0.02;
  int a_subset = 2000;
  uint64_t max_height = 0;        // 0 = mine forever
  uint16_t api_port = 0;          // 0 = no HTTP API
  double blob_timeout_s = 10.0;   // per peer, when fetching a missing checkpoint
};

Config parse_args(int argc, char** argv);

class Node {
public:
  explicit Node(Config cfg);
  void run();  // blocks: mining loop on the calling thread

  // For the HTTP API; thread-safe.
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

  // Full verification: structure + B (+ A). Returns the rejection reason or nullopt. All checkpoints are
  // fetched from peers before the transitions are picked; if they cannot be fetched, `unavailable` is set.
  std::optional<std::string> validate(int peer, const Block& b, const Block& parent, std::string& detail,
                                      json& checks, bool& unavailable);
  // Makes sure the store has every blob, asking `peer` first and then the others. On failure returns
  // the hash that could not be fetched.
  std::optional<Hash> fetch_blobs(int peer, const std::vector<Hash>& hashes);
  std::string blob_path(const Hash& h) const;
  void serve_blob(int peer, const json& msg);
  void receive_blob(int peer, const json& msg);
  json status_locked() const;  // under mu_
  json worker_task(const char* op, const Block& b) const;
  void persist();  // under mu_
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
  std::map<Hash, Incoming> orphans_;  // parent not known yet
  std::map<int, std::chrono::steady_clock::time_point> last_sync_;
  bool validating_ = false;  // validator is checking a block right now (under mu_)
  std::atomic<bool> tip_changed_{false};
  std::atomic<bool> lazy_mode_{false};  // toggled via the API: the node becomes a lazy attacker
  std::atomic<bool> paused_{false};
  std::map<int, std::string> peer_ids_;  // peer → node_id from hello (under mu_)

  struct BlobReply {
    bool missing = false;
    std::string tmp_path;  // received bytes, not yet checked against the hash
  };
  struct BlobRequest {
    int peer;       // only this peer's answer counts
    uint64_t seq;   // names the temporary file, so a late answer never overwrites a newer one
    std::optional<BlobReply> reply;
  };
  std::mutex blob_mu_;
  std::condition_variable blob_cv_;
  std::map<Hash, BlobRequest> blob_wait_;  // outstanding requests (under blob_mu_)
  uint64_t blob_seq_ = 0;                  // under blob_mu_
  // Blocks skipped for unavailable data, with the time: a resend within the cooldown is ignored, so a miner
  // that never serves its checkpoints cannot keep the validator waiting on timeouts. Validator thread only.
  std::map<Hash, std::chrono::steady_clock::time_point> skipped_;
  std::atomic<uint64_t> blobs_fetched_{0}, blobs_served_{0};
  std::mt19937_64 rng_{std::random_device{}()};  // private choice of transitions (D5)
  uint64_t mined_ = 0, accepted_ = 0, rejected_count_ = 0, lost_races_ = 0;
};

}  // namespace pouw
