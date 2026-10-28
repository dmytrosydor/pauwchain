#include "node.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace pouw {

namespace {

using Clock = std::chrono::steady_clock;

std::mutex g_log_mu;

std::string short_hex(const Hash& h) { return to_hex(h).substr(0, 10); }

std::string fmt_double(double v) {
  std::ostringstream s;
  s.precision(3);
  s << std::scientific << v;
  return s.str();
}

uint64_t unix_now() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

// Sleep that is interrupted by a tip change.
bool sleep_unless(const std::atomic<bool>& flag, double seconds) {
  auto until = Clock::now() + std::chrono::duration<double>(seconds);
  while (Clock::now() < until) {
    if (flag) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return true;
}

std::pair<std::string, uint16_t> parse_host_port(const std::string& s) {
  auto colon = s.rfind(':');
  if (colon == std::string::npos) throw std::invalid_argument("peer must be host:port: " + s);
  return {s.substr(0, colon), static_cast<uint16_t>(std::stoi(s.substr(colon + 1)))};
}

}  // namespace

Config parse_args(int argc, char** argv) {
  Config c;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--id") c.id = need(i);
    else if (a == "--port") c.port = static_cast<uint16_t>(std::stoi(need(i)));
    else if (a == "--listen") c.listen = need(i);
    else if (a == "--api-listen") c.api_listen = need(i);
    else if (a == "--peers") {
      std::stringstream ss(need(i));
      for (std::string p; std::getline(ss, p, ',');) {
        if (!p.empty()) c.peers.push_back(parse_host_port(p));
      }
    } else if (a == "--store") c.store_dir = need(i);
    else if (a == "--task") c.task_path = need(i);
    else if (a == "--python") c.python = need(i);
    else if (a == "--worker") c.worker_script = need(i);
    else if (a == "--worker-mode") c.worker_mode = need(i);
    else if (a == "--run-dir") c.run_dir = need(i);
    else if (a == "--device") c.device = need(i);
    else if (a == "--lazy") c.lazy = true;
    else if (a == "--lazy-delay") c.lazy_delay_s = std::stod(need(i));
    else if (a == "--q") c.q = std::stoi(need(i));
    else if (a == "--alpha") c.alpha = std::stod(need(i));
    else if (a == "--verify-a") c.verify_a = true;
    else if (a == "--max-height") c.max_height = std::stoull(need(i));
    else if (a == "--api-port") c.api_port = static_cast<uint16_t>(std::stoi(need(i)));
    else if (a == "--blob-timeout") c.blob_timeout_s = std::stod(need(i));
    else throw std::invalid_argument("unknown argument: " + a);
  }
  if (c.id.empty() || c.port == 0 || c.task_path.empty()) {
    throw std::invalid_argument("required: --id, --port, --task");
  }
  if (c.run_dir.empty()) c.run_dir = "run/node-" + c.id;
  if (c.worker_mode != "server" && c.worker_mode != "spawn") {
    throw std::invalid_argument("--worker-mode must be server or spawn");
  }
  return c;
}

Node::Node(Config cfg) : cfg_(std::move(cfg)) {
  std::ifstream in(cfg_.task_path);
  if (!in) throw std::runtime_error("cannot read task file " + cfg_.task_path);
  std::string raw((std::istreambuf_iterator<char>(in)), {});
  task_ = task_from_json(json::parse(raw), raw);
  miner_id_ = sha256("node-" + cfg_.id);
  std::filesystem::create_directories(cfg_.run_dir);
  if (cfg_.worker_mode == "server") {
    // Separate processes for mining and verification: checking others' blocks does not wait for our own training.
    miner_bridge_ = std::make_unique<WorkerBridge>(cfg_.python, cfg_.worker_server, cfg_.run_dir + "/mine",
                                                   cfg_.run_dir + "/mine.sock", task_.dataset);
    verifier_bridge_ = std::make_unique<WorkerBridge>(cfg_.python, cfg_.worker_server,
                                                      cfg_.run_dir + "/verify", cfg_.run_dir + "/verify.sock",
                                                      task_.dataset);
  } else {
    miner_bridge_ = std::make_unique<WorkerBridge>(cfg_.python, cfg_.worker_script, cfg_.run_dir + "/mine");
    verifier_bridge_ = std::make_unique<WorkerBridge>(cfg_.python, cfg_.worker_script,
                                                      cfg_.run_dir + "/verify");
  }
  lazy_mode_ = cfg_.lazy;
  events_ = std::make_unique<EventBus>(cfg_.run_dir + "/events.jsonl");
}

void Node::log(const std::string& line) const {
  std::lock_guard lock(g_log_mu);
  std::cout << "[node " << cfg_.id << "] " << line << std::endl;
}

void Node::run() {
  auto g = miner_bridge_->run(
      {{"op", "genesis"}, {"task_seed", task_.task_seed}, {"store_dir", cfg_.store_dir}});
  if (!g || !g->value("ok", false)) {
    throw std::runtime_error("genesis failed: " + (g ? g->dump() : std::string("cancelled")));
  }
  Hash w0 = hash_from_hex(g->at("w_start").get<std::string>());
  chain_ = std::make_unique<Chain>(make_genesis(task_, w0));
  size_t restored = chain_->load(cfg_.run_dir + "/chain.json");
  log("genesis " + short_hex(chain_->genesis().hash()) + " w0=" + short_hex(w0) +
      (restored ? ", restored " + std::to_string(restored) + " blocks" : "") +
      (cfg_.lazy ? ", mode=LAZY" : ""));
  {
    std::lock_guard lock(mu_);
    persist();
  }
  events_->publish("genesis", {{"hash", to_hex(chain_->genesis().hash())}, {"w0", to_hex(w0)},
                               {"height", chain_->height()}});

  if (cfg_.api_port) {
    ApiHandlers h;
    h.status = [this] { return status(); };
    h.chain_from = [this](uint64_t from) { return chain_from(from); };
    h.block = [this](const std::string& hex) { return block_json(hex); };
    h.admin = [this](const json& req) { return admin(req); };
    h.events = events_.get();
    api_ = std::make_unique<Api>(cfg_.api_listen, cfg_.api_port, std::move(h));
    api_->start();
    log("API on http://" + cfg_.api_listen + ":" + std::to_string(cfg_.api_port));
  }

  // Leftovers from an interrupted fetch; nothing in here has passed the hash check.
  std::filesystem::remove_all(cfg_.store_dir + "/incoming");
  std::filesystem::create_directories(cfg_.store_dir + "/incoming");
  p2p_ = std::make_unique<P2P>(
      cfg_.listen, cfg_.port, [this](int peer) { on_connect(peer); },
      [this](int peer, const json& msg) { on_message(peer, msg); });
  p2p_->start();
  for (const auto& [host, port] : cfg_.peers) p2p_->connect_to(host, port);
  std::thread([this] { validator_loop(); }).detach();

  // Give peers time to connect so everyone starts from the same state.
  auto deadline = Clock::now() + std::chrono::seconds(5);
  while (p2p_->peer_count() < cfg_.peers.size() && Clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  mining_loop();
  log("reached max height, mining stopped");
  for (;;) std::this_thread::sleep_for(std::chrono::seconds(60));
}

json Node::worker_task(const char* op, const Block& b) const {
  return {
      {"op", op},
      {"task_id", to_hex(b.header.task_id)},
      {"round", b.header.round},
      {"miner_id", to_hex(b.header.miner_id)},
      {"prev_block_hash", to_hex(b.header.prev_hash)},
      {"batch_size", task_.batch_size},
      {"lr", task_.lr},
      {"dataset", task_.dataset},
      {"store_dir", cfg_.store_dir},
      {"device", cfg_.device},
  };
}

void Node::mining_loop() {
  for (;;) {
    // Process received blocks first: otherwise the node mines on a stale tip and spawns forks.
    // The wait limit keeps attacker spam from stalling mining completely.
    auto wait_until = Clock::now() + std::chrono::seconds(20);
    for (;;) {
      {
        std::lock_guard lock(mu_);
        if ((inbox_.empty() && !validating_) || Clock::now() > wait_until) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (paused_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    const bool lazy = lazy_mode_;
    Block parent;
    {
      std::lock_guard lock(mu_);
      if (cfg_.max_height && chain_->height() >= cfg_.max_height) return;
      parent = chain_->tip();
      tip_changed_ = false;
    }

    Block b;
    Header& h = b.header;
    h.height = parent.header.height + 1;
    h.prev_hash = parent.hash();
    h.task_id = task_.task_id;
    h.round = h.height;
    h.miner_id = miner_id_;
    h.w_start = parent.header.w_end;
    h.K = task_.K;
    h.k = task_.k;

    if (lazy) {
      if (!sleep_unless(tip_changed_, cfg_.lazy_delay_s)) continue;
      // Lazy replay: no training steps, every checkpoint is a copy of the starting weights.
      b.ckpt_hashes.assign(task_.K / task_.k + 1, h.w_start);
      h.w_end = h.w_start;
      h.metric_micro = parent.header.metric_micro;
    } else {
      json t = worker_task("train", b);
      t["w_start"] = to_hex(h.w_start);
      t["K"] = task_.K;
      t["k"] = task_.k;
      auto res = miner_bridge_->run(t, &tip_changed_);
      if (!res) {
        log("mining aborted at height " + std::to_string(h.height) + ": new tip");
        events_->publish("mining_aborted", {{"height", h.height}});
        continue;
      }
      if (!res->value("ok", false)) {
        log("train failed: " + res->value("error", std::string("?")));
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      for (const auto& c : res->at("ckpt_hashes")) {
        b.ckpt_hashes.push_back(hash_from_hex(c.get<std::string>()));
      }
      h.w_end = hash_from_hex(res->at("w_end").get<std::string>());
      h.metric_micro = res->at("metric_micro").get<int64_t>();
    }
    h.ckpt_root = ckpt_root_of(b.ckpt_hashes);
    h.timestamp = unix_now();

    {
      std::lock_guard lock(mu_);
      if (chain_->tip().hash() != parent.hash()) {
        ++lost_races_;
        log("lost race at height " + std::to_string(h.height) + ", block discarded");
        events_->publish("lost_race", {{"height", h.height}});
        continue;
      }
      ++mined_;
      // The attacker does not add its own blocks locally; it follows the honest chain to stay in the race.
      if (!lazy) chain_->add(b);
      persist();
    }
    log(std::string(lazy ? "LAZY block" : "MINED") + " h=" + std::to_string(h.height) +
        " hash=" + short_hex(b.hash()) + " acc=" + std::to_string(h.metric_micro / 1e4) + "%");
    events_->publish("mined", {{"height", h.height}, {"hash", to_hex(b.hash())}, {"lazy", lazy},
                               {"metric_micro", h.metric_micro}});
    if (!lazy) {
      events_->publish("tip", {{"height", h.height}, {"hash", to_hex(b.hash())},
                               {"metric_micro", h.metric_micro}, {"miner", to_hex(h.miner_id)}});
    }
    p2p_->broadcast({{"type", "block"}, {"block", block_to_json(b)}});
  }
}

void Node::on_connect(int peer) {
  uint64_t height;
  {
    std::lock_guard lock(mu_);
    height = chain_->height();
  }
  p2p_->send(peer, {{"type", "hello"}, {"node_id", cfg_.id}, {"height", height}});
}

void Node::request_full_sync(int peer) {
  auto now = Clock::now();
  auto it = last_sync_.find(peer);
  if (it != last_sync_.end() && now - it->second < std::chrono::seconds(3)) return;
  last_sync_[peer] = now;
  p2p_->send(peer, {{"type", "get_blocks"}, {"from_height", 1}});
}

void Node::on_message(int peer, const json& msg) {
  const std::string type = msg.at("type").get<std::string>();
  if (type == "hello") {
    uint64_t ours;
    {
      std::lock_guard lock(mu_);
      ours = chain_->height();
      peer_ids_[peer] = msg.value("node_id", std::string("?"));
    }
    events_->publish("peer", {{"peer", peer}, {"node_id", msg.value("node_id", std::string("?"))},
                              {"height", msg.at("height")}});
    if (msg.at("height").get<uint64_t>() > ours) {
      p2p_->send(peer, {{"type", "get_blocks"}, {"from_height", ours + 1}});
    }
  } else if (type == "get_blocks") {
    json blocks = json::array();
    {
      std::lock_guard lock(mu_);
      for (const auto& b : chain_->best_from(msg.at("from_height").get<uint64_t>())) {
        blocks.push_back(block_to_json(b));
      }
    }
    p2p_->send(peer, {{"type", "blocks"}, {"blocks", blocks}});
  } else if (type == "get_blob") {
    serve_blob(peer, msg);
  } else if (type == "blob") {
    receive_blob(peer, msg);
  } else if (type == "block" || type == "blocks") {
    std::vector<Block> got;
    if (type == "block") {
      got.push_back(block_from_json(msg.at("block")));
    } else {
      for (const auto& j : msg.at("blocks")) got.push_back(block_from_json(j));
    }
    {
      std::lock_guard lock(mu_);
      for (auto& b : got) inbox_.push_back({peer, std::move(b)});
    }
    cv_.notify_one();
  }
}

std::string Node::blob_path(const Hash& h) const {
  return cfg_.store_dir + "/" + to_hex(h) + ".safetensors";
}

void Node::serve_blob(int peer, const json& msg) {
  const Hash h = hash_from_hex(msg.at("hash").get<std::string>());
  json reply = {{"type", "blob"}, {"hash", to_hex(h)}};
  std::ifstream in(blob_path(h), std::ios::binary);
  std::vector<uint8_t> bytes;
  if (in) bytes.assign(std::istreambuf_iterator<char>(in), {});
  if (bytes.empty()) {
    reply["missing"] = true;
  } else {
    reply["data"] = to_base64(bytes);
    ++blobs_served_;
  }
  p2p_->send(peer, reply);
}

void Node::receive_blob(int peer, const json& msg) {
  const Hash h = hash_from_hex(msg.at("hash").get<std::string>());
  uint64_t seq;
  {
    std::lock_guard lock(blob_mu_);
    auto it = blob_wait_.find(h);
    // Not requested, asked from another peer, or already answered.
    if (it == blob_wait_.end() || it->second.peer != peer || it->second.reply) return;
    seq = it->second.seq;
  }
  BlobReply reply;
  reply.missing = true;  // also for a malformed answer: the validator moves on to the next peer at once
  if (!msg.value("missing", false) && msg.contains("data") && msg.at("data").is_string()) {
    try {
      auto bytes = from_base64(msg.at("data").get<std::string>());
      reply.tmp_path = cfg_.store_dir + "/incoming/" + to_hex(h) + "." + std::to_string(seq) + ".tmp";
      std::ofstream(reply.tmp_path, std::ios::binary)
          .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      reply.missing = false;
    } catch (const std::invalid_argument&) {
    }
  }
  std::lock_guard lock(blob_mu_);
  auto it = blob_wait_.find(h);
  if (it == blob_wait_.end() || it->second.seq != seq || it->second.reply) {
    if (!reply.tmp_path.empty()) std::filesystem::remove(reply.tmp_path);  // the request timed out meanwhile
    return;
  }
  it->second.reply = std::move(reply);
  blob_cv_.notify_all();
}

std::optional<Hash> Node::fetch_blobs(int peer, const std::vector<Hash>& hashes) {
  for (const Hash& h : hashes) {
    if (std::filesystem::exists(blob_path(h))) continue;
    // The sender first, if still connected, then everyone else.
    std::vector<int> order, connected = p2p_->peer_ids();
    if (std::find(connected.begin(), connected.end(), peer) != connected.end()) order.push_back(peer);
    for (int p : connected) {
      if (p != peer) order.push_back(p);
    }
    bool got = false;
    for (int from : order) {
      {
        std::lock_guard lock(blob_mu_);
        blob_wait_[h] = {from, ++blob_seq_, std::nullopt};
      }
      p2p_->send(from, {{"type", "get_blob"}, {"hash", to_hex(h)}});
      std::optional<BlobReply> reply;
      {
        std::unique_lock lock(blob_mu_);
        blob_cv_.wait_for(lock, std::chrono::duration<double>(cfg_.blob_timeout_s),
                          [&] { return blob_wait_[h].reply.has_value(); });
        reply = std::move(blob_wait_[h].reply);
        blob_wait_.erase(h);
      }
      if (!reply || reply->missing) continue;
      // The worker checks the canonical hash before the file enters the store; a bad copy is deleted.
      auto res = verifier_bridge_->run(
          {{"op", "import"}, {"store_dir", cfg_.store_dir}, {"path", reply->tmp_path}, {"hash", to_hex(h)}});
      if (res && res->value("ok", false) && res->value("valid", false)) {
        ++blobs_fetched_;
        got = true;
        break;
      }
      log("blob " + short_hex(h) + " from peer " + std::to_string(from) + " does not match its hash");
    }
    if (!got) return h;
  }
  return std::nullopt;
}

std::optional<std::string> Node::validate(int peer, const Block& b, const Block& parent,
                                          std::string& detail, json& checks, bool& unavailable) {
  if (auto err = check_structure(b, parent, task_)) return "structure: " + *err;

  // Every checkpoint is fetched before any transition is picked. Asking only for the picked ones would tell
  // the miner the private choice (D5): it could withhold the data of a bad transition and resend the block
  // (with a new timestamp, so a new hash) until a draw misses it.
  if (auto missing = fetch_blobs(peer, b.ckpt_hashes)) {
    unavailable = true;
    return "data unavailable: checkpoint " + short_hex(*missing);
  }

  // Mechanism B: q transitions chosen by the validator's private RNG.
  const size_t m = b.header.K / b.header.k;
  std::vector<size_t> all(m);
  for (size_t j = 0; j < m; ++j) all[j] = j;
  std::vector<size_t> picks;
  std::sample(all.begin(), all.end(), std::back_inserter(picks),
              std::min<size_t>(static_cast<size_t>(cfg_.q), m), rng_);

  // Transitions are independent, so they are recomputed in parallel, each in its own worker process.
  std::vector<std::future<std::optional<json>>> runs;
  for (size_t j : picks) {
    json t = worker_task("verify_b", b);
    t["ckpt_from"] = to_hex(b.ckpt_hashes[j]);
    t["ckpt_to"] = to_hex(b.ckpt_hashes[j + 1]);
    t["step_from"] = j * b.header.k;
    t["k"] = b.header.k;
    t["threshold_rule"] = {{"kind", "relative"}, {"alpha", cfg_.alpha}};
    runs.push_back(std::async(std::launch::async, [this, t] { return verifier_bridge_->run(t); }));
  }
  for (size_t i = 0; i < picks.size(); ++i) {
    size_t j = picks[i];
    auto res = runs[i].get();
    if (!res || !res->value("ok", false)) {
      return "B: worker error: " + (res ? res->value("error", std::string("?")) : "cancelled");
    }
    detail += " B[j=" + std::to_string(j) + " d=" + fmt_double(res->at("distance").get<double>()) +
              " thr=" + fmt_double(res->at("threshold").get<double>()) + "]";
    checks.push_back({{"mech", "B"}, {"j", j}, {"distance", res->at("distance")},
                      {"threshold", res->at("threshold")}, {"verdict", res->at("verdict")}});
    if (res->at("verdict") != "accept") return "B: transition " + std::to_string(j) + " failed";
  }

  if (cfg_.verify_a) {
    json t = worker_task("verify_a", b);
    t["w_end"] = to_hex(b.header.w_end);
    t["metric_micro"] = b.header.metric_micro;
    t["subset"] = cfg_.a_subset;
    t["eps"] = cfg_.a_eps;
    auto res = verifier_bridge_->run(t);
    if (!res || !res->value("ok", false)) {
      return "A: worker error: " + (res ? res->value("error", std::string("?")) : "cancelled");
    }
    detail += " A[measured=" + std::to_string(res->at("measured").get<double>()) + "]";
    checks.push_back({{"mech", "A"}, {"measured", res->at("measured")}, {"claimed", res->at("claimed")},
                      {"verdict", res->at("verdict")}});
    if (res->at("verdict") != "accept") return "A: claimed metric not reproduced";
  }
  return std::nullopt;
}

void Node::validator_loop() {
  for (;;) {
    Incoming in;
    Block parent;
    Hash h;
    {
      std::unique_lock lock(mu_);
      cv_.wait(lock, [this] { return !inbox_.empty(); });
      in = std::move(inbox_.front());
      inbox_.pop_front();
      h = in.block.hash();
      if (chain_->contains(h) || rejected_.count(h)) continue;
      if (auto it = skipped_.find(h); it != skipped_.end()) {
        if (Clock::now() - it->second < std::chrono::seconds(30)) continue;
        skipped_.erase(it);
      }
      const Hash& prev = in.block.header.prev_hash;
      if (rejected_.count(prev)) {
        rejected_.insert(h);
        ++rejected_count_;
        log("REJECT h=" + std::to_string(in.block.header.height) + " hash=" + short_hex(h) +
            ": parent was rejected");
        events_->publish("verdict", {{"height", in.block.header.height}, {"hash", to_hex(h)},
                                     {"miner", to_hex(in.block.header.miner_id)}, {"accepted", false},
                                     {"reason", "parent was rejected"}, {"checks", json::array()}});
        persist();
        continue;
      }
      if (!chain_->contains(prev)) {
        orphans_.emplace(h, in);
        if (orphans_.size() > 256) orphans_.erase(orphans_.begin());
        request_full_sync(in.peer);
        continue;
      }
      parent = *chain_->get(prev);
      validating_ = true;
    }

    std::string detail;
    json checks = json::array();
    auto t0 = Clock::now();
    bool unavailable = false;
    auto reason = validate(in.peer, in.block, parent, detail, checks, unavailable);
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();

    std::string what = "h=" + std::to_string(in.block.header.height) + " hash=" + short_hex(h) +
                       " miner=" + short_hex(in.block.header.miner_id);
    {
      std::lock_guard lock(mu_);
      validating_ = false;
      if (unavailable) {
        // Not a verdict: the block may be fine, and a resend or sync after the cooldown will retry it.
        if (skipped_.size() >= 1024) {
          std::erase_if(skipped_, [](const auto& e) { return Clock::now() - e.second > std::chrono::seconds(30); });
        }
        if (skipped_.size() < 1024) skipped_[h] = Clock::now();
        log("SKIP " + what + ": " + *reason);
        events_->publish("unavailable", {{"height", in.block.header.height}, {"hash", to_hex(h)},
                                         {"miner", to_hex(in.block.header.miner_id)}, {"reason", *reason}});
        continue;
      }
      json verdict = {{"height", in.block.header.height}, {"hash", to_hex(h)},
                      {"miner", to_hex(in.block.header.miner_id)}, {"accepted", !reason},
                      {"reason", reason ? *reason : ""}, {"checks", checks}, {"seconds", secs},
                      {"metric_micro", in.block.header.metric_micro}};
      if (reason) {
        rejected_.insert(h);
        ++rejected_count_;
        log("REJECT " + what + ": " + *reason + detail);
        events_->publish("verdict", verdict);
      } else {
        bool new_tip = chain_->add(in.block);
        ++accepted_;
        if (new_tip) tip_changed_ = true;
        log("ACCEPT " + what + detail + " (" + std::to_string(secs).substr(0, 4) + "s)" +
            (new_tip ? "" : " [side branch]"));
        verdict["new_tip"] = new_tip;
        events_->publish("verdict", verdict);
        if (new_tip) {
          const Block& t = chain_->tip();
          events_->publish("tip", {{"height", t.header.height}, {"hash", to_hex(t.hash())},
                                   {"metric_micro", t.header.metric_micro},
                                   {"miner", to_hex(t.header.miner_id)}});
        }
      }
      // This block's children can now be processed: either verified or rejected along with the parent.
      for (auto it = orphans_.begin(); it != orphans_.end();) {
        if (it->second.block.header.prev_hash == h) {
          inbox_.push_front(std::move(it->second));
          it = orphans_.erase(it);
        } else {
          ++it;
        }
      }
      persist();
    }
    if (!reason) p2p_->broadcast({{"type", "block"}, {"block", block_to_json(in.block)}}, in.peer);
  }
}

json Node::status_locked() const {
  json peers = json::array();
  for (const auto& [peer, id] : peer_ids_) peers.push_back({{"peer", peer}, {"node_id", id}});
  const Block& tip = chain_->tip();
  return {
      {"id", cfg_.id},
      {"miner_id", to_hex(miner_id_)},
      {"lazy", lazy_mode_.load()},
      {"paused", paused_.load()},
      {"height", chain_->height()},
      {"tip", to_hex(tip.hash())},
      {"tip_metric_micro", tip.header.metric_micro},
      {"mined", mined_},
      {"accepted", accepted_},
      {"rejected", rejected_count_},
      {"lost_races", lost_races_},
      {"blobs_fetched", blobs_fetched_.load()},
      {"blobs_served", blobs_served_.load()},
      {"device", cfg_.device},
      {"q", cfg_.q},
      {"alpha", cfg_.alpha},
      {"verify_a", cfg_.verify_a},
      {"task", {{"K", task_.K}, {"k", task_.k}, {"dataset", task_.dataset}}},
      {"peers", peers},
      {"last_event", events_->last_seq()},
  };
}

json Node::status() {
  std::lock_guard lock(mu_);
  return status_locked();
}

json Node::chain_from(uint64_t from_height) {
  std::lock_guard lock(mu_);
  json blocks = json::array();
  for (const auto& b : chain_->best_from(from_height)) {
    json j = block_to_json(b);
    j["hash"] = to_hex(b.hash());
    blocks.push_back(std::move(j));
  }
  return {{"height", chain_->height()}, {"blocks", blocks}};
}

std::optional<json> Node::block_json(const std::string& hash_hex) {
  std::lock_guard lock(mu_);
  const Block* b = chain_->get(hash_from_hex(hash_hex));
  if (!b) return std::nullopt;
  json j = block_to_json(*b);
  j["hash"] = hash_hex;
  return j;
}

json Node::admin(const json& request) {
  if (request.contains("mode")) {
    const std::string mode = request.at("mode").get<std::string>();
    if (mode != "honest" && mode != "lazy") return {{"error", "mode must be honest or lazy"}};
    lazy_mode_ = mode == "lazy";
    tip_changed_ = true;  // interrupt the current round so the new mode applies immediately
    log(std::string("mode switched to ") + (lazy_mode_ ? "LAZY" : "honest"));
  }
  if (request.contains("paused")) {
    paused_ = request.at("paused").get<bool>();
    tip_changed_ = true;
    log(paused_ ? "mining paused" : "mining resumed");
  }
  events_->publish("mode", {{"lazy", lazy_mode_.load()}, {"paused", paused_.load()}});
  std::lock_guard lock(mu_);
  persist();
  return status_locked();
}

void Node::persist() {
  chain_->save(cfg_.run_dir + "/chain.json");
  std::string tmp = cfg_.run_dir + "/status.json.tmp";
  std::ofstream(tmp) << status_locked().dump(2) << '\n';
  std::filesystem::rename(tmp, cfg_.run_dir + "/status.json");
}

}  // namespace pouw
