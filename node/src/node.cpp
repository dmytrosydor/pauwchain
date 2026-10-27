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

// Сон, що переривається зміною вершини.
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
    else if (a == "--peers") {
      std::stringstream ss(need(i));
      for (std::string p; std::getline(ss, p, ',');) {
        if (!p.empty()) c.peers.push_back(parse_host_port(p));
      }
    } else if (a == "--store") c.store_dir = need(i);
    else if (a == "--task") c.task_path = need(i);
    else if (a == "--python") c.python = need(i);
    else if (a == "--worker") c.worker_script = need(i);
    else if (a == "--run-dir") c.run_dir = need(i);
    else if (a == "--device") c.device = need(i);
    else if (a == "--lazy") c.lazy = true;
    else if (a == "--lazy-delay") c.lazy_delay_s = std::stod(need(i));
    else if (a == "--q") c.q = std::stoi(need(i));
    else if (a == "--alpha") c.alpha = std::stod(need(i));
    else if (a == "--verify-a") c.verify_a = true;
    else if (a == "--max-height") c.max_height = std::stoull(need(i));
    else throw std::invalid_argument("unknown argument: " + a);
  }
  if (c.id.empty() || c.port == 0 || c.task_path.empty()) {
    throw std::invalid_argument("required: --id, --port, --task");
  }
  if (c.run_dir.empty()) c.run_dir = "run/node-" + c.id;
  return c;
}

Node::Node(Config cfg) : cfg_(std::move(cfg)) {
  std::ifstream in(cfg_.task_path);
  if (!in) throw std::runtime_error("cannot read task file " + cfg_.task_path);
  std::string raw((std::istreambuf_iterator<char>(in)), {});
  task_ = task_from_json(json::parse(raw), raw);
  miner_id_ = sha256("node-" + cfg_.id);
  std::filesystem::create_directories(cfg_.run_dir);
  miner_bridge_ = std::make_unique<WorkerBridge>(cfg_.python, cfg_.worker_script,
                                                 cfg_.run_dir + "/mine");
  verifier_bridge_ = std::make_unique<WorkerBridge>(cfg_.python, cfg_.worker_script,
                                                    cfg_.run_dir + "/verify");
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

  p2p_ = std::make_unique<P2P>(
      cfg_.port, [this](int peer) { on_connect(peer); },
      [this](int peer, const json& msg) { on_message(peer, msg); });
  p2p_->start();
  for (const auto& [host, port] : cfg_.peers) p2p_->connect_to(host, port);
  std::thread([this] { validator_loop(); }).detach();

  // Даємо пірам підключитися, щоб усі стартували з одного стану.
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
    // Спершу дообробити отримані блоки: інакше вузол майнить на застарілій вершині й плодить форки.
    // Обмеження очікування не дає спаму від атакувальника зупинити майнінг повністю.
    auto wait_until = Clock::now() + std::chrono::seconds(20);
    for (;;) {
      {
        std::lock_guard lock(mu_);
        if ((inbox_.empty() && !validating_) || Clock::now() > wait_until) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

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

    if (cfg_.lazy) {
      if (!sleep_unless(tip_changed_, cfg_.lazy_delay_s)) continue;
      // Ліниве повторення: жодного кроку навчання, усі точки — копії стартових ваг.
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
        continue;
      }
      // Атакувальник не додає свої блоки собі, а йде за чесним ланцюгом, щоб лишатися в перегонах.
      if (!cfg_.lazy) {
        chain_->add(b);
        persist();
      }
      ++mined_;
    }
    log(std::string(cfg_.lazy ? "LAZY block" : "MINED") + " h=" + std::to_string(h.height) +
        " hash=" + short_hex(b.hash()) + " acc=" + std::to_string(h.metric_micro / 1e4) + "%");
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
    }
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

std::optional<std::string> Node::validate(const Block& b, const Block& parent,
                                          std::string& detail) {
  if (auto err = check_structure(b, parent, task_)) return "structure: " + *err;

  // Механізм B: q переходів, обраних приватним генератором валідатора.
  const size_t m = b.header.K / b.header.k;
  std::vector<size_t> all(m);
  for (size_t j = 0; j < m; ++j) all[j] = j;
  std::vector<size_t> picks;
  std::sample(all.begin(), all.end(), std::back_inserter(picks),
              std::min<size_t>(static_cast<size_t>(cfg_.q), m), rng_);

  // Переходи незалежні, тож перераховуються паралельно, кожен у своєму процесі воркера.
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
      const Hash& prev = in.block.header.prev_hash;
      if (rejected_.count(prev)) {
        rejected_.insert(h);
        ++rejected_count_;
        log("REJECT h=" + std::to_string(in.block.header.height) + " hash=" + short_hex(h) +
            ": parent was rejected");
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
    auto t0 = Clock::now();
    auto reason = validate(in.block, parent, detail);
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();

    std::string what = "h=" + std::to_string(in.block.header.height) + " hash=" + short_hex(h) +
                       " miner=" + short_hex(in.block.header.miner_id);
    {
      std::lock_guard lock(mu_);
      validating_ = false;
      if (reason) {
        rejected_.insert(h);
        ++rejected_count_;
        log("REJECT " + what + ": " + *reason + detail);
      } else {
        bool new_tip = chain_->add(in.block);
        ++accepted_;
        if (new_tip) tip_changed_ = true;
        log("ACCEPT " + what + detail + " (" + std::to_string(secs).substr(0, 4) + "s)" +
            (new_tip ? "" : " [side branch]"));
      }
      // Діти цього блоку тепер можна обробити: або перевірити, або відхилити разом із батьком.
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

void Node::persist() {
  chain_->save(cfg_.run_dir + "/chain.json");
  json status = {
      {"id", cfg_.id},
      {"lazy", cfg_.lazy},
      {"height", chain_->height()},
      {"tip", to_hex(chain_->tip().hash())},
      {"mined", mined_},
      {"accepted", accepted_},
      {"rejected", rejected_count_},
      {"lost_races", lost_races_},
  };
  std::string tmp = cfg_.run_dir + "/status.json.tmp";
  std::ofstream(tmp) << status.dump(2) << '\n';
  std::filesystem::rename(tmp, cfg_.run_dir + "/status.json");
}

}  // namespace pouw
