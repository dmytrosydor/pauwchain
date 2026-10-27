#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <string>

#include "block.hpp"
#include "chain.hpp"
#include "events.hpp"

using namespace pouw;

static int g_failed = 0;

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::cerr << __FILE__ << ":" << __LINE__ << ": FAIL " #cond "\n"; \
      ++g_failed;                                                        \
    }                                                                    \
  } while (0)

static Hash h_of(uint8_t byte) {
  Hash h;
  h.fill(byte);
  return h;
}

static Task test_task() {
  Task t;
  t.task_id = h_of(0x11);
  t.K = 100;
  t.k = 25;
  return t;
}

// Honest-looking child block: m+1 checkpoints, C₀ = parent's w_end.
static Block child_of(const Block& parent, uint8_t salt) {
  Block b;
  Header& h = b.header;
  h.height = parent.header.height + 1;
  h.prev_hash = parent.hash();
  h.task_id = parent.header.task_id;
  h.round = h.height;
  h.miner_id = h_of(salt);
  h.w_start = parent.header.w_end;
  h.K = 100;
  h.k = 25;
  b.ckpt_hashes.push_back(h.w_start);
  for (uint8_t i = 1; i <= 4; ++i) b.ckpt_hashes.push_back(h_of(static_cast<uint8_t>(salt + i)));
  h.w_end = b.ckpt_hashes.back();
  h.ckpt_root = ckpt_root_of(b.ckpt_hashes);
  h.metric_micro = 900000;
  return b;
}

static void test_hex() {
  Hash h = h_of(0xab);
  CHECK(hash_from_hex(to_hex(h)) == h);
  CHECK(to_hex(sha256(std::string_view("abc"))).rfind("ba7816bf", 0) == 0);
}

static void test_golden(const std::string& path) {
  std::ifstream in(path);
  CHECK(static_cast<bool>(in));
  if (!in) return;
  json g = json::parse(in);
  Header h = header_from_json(g.at("header"));
  CHECK(h.serialize().size() == kHeaderSize);
  CHECK(to_hex(h.hash()) == g.at("block_hash").get<std::string>());
  CHECK(header_to_json(h) == g.at("header"));
}

// Vectors shared with Python (scripts/gen_vectors.py): header bytes, block hash, ckpt_root.
static void test_vectors(const std::string& path) {
  std::ifstream in(path);
  CHECK(static_cast<bool>(in));
  if (!in) return;
  json v = json::parse(in);
  for (const auto& c : v.at("headers")) {
    Header h = header_from_json(c.at("header"));
    auto raw = h.serialize();
    CHECK(to_hex(raw) == c.at("bytes_hex").get<std::string>());
    CHECK(to_hex(h.hash()) == c.at("block_hash").get<std::string>());
  }
  for (const auto& c : v.at("ckpt_roots")) {
    std::vector<Hash> hs;
    for (const auto& x : c.at("hashes")) hs.push_back(hash_from_hex(x.get<std::string>()));
    CHECK(to_hex(ckpt_root_of(hs)) == c.at("root").get<std::string>());
  }
}

// The tip does not depend on the order in which the node received the tree's blocks.
static void test_chain_order_independent() {
  Task t = test_task();
  Block g = make_genesis(t, h_of(0x33));
  std::vector<Block> blocks;
  std::vector<Block> layer{g};
  uint8_t salt = 0x40;
  for (int depth = 0; depth < 3; ++depth) {
    std::vector<Block> next;
    for (const auto& parent : layer)
      for (int i = 0; i < 2; ++i) {
        next.push_back(child_of(parent, salt));
        salt = static_cast<uint8_t>(salt + 7);
      }
    blocks.insert(blocks.end(), next.begin(), next.end());
    layer = next;
  }
  Chain ref(g);
  for (const auto& b : blocks) ref.add(b);
  std::mt19937 rng(7);
  for (int trial = 0; trial < 50; ++trial) {
    std::vector<Block> shuffled = blocks;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    Chain c(g);
    // Chain rejects a block with an unknown parent (throws); the node fetches the parent and redelivers.
    for (int pass = 0; pass < 4; ++pass)
      for (const auto& b : shuffled) {
        try {
          c.add(b);
        } catch (const std::invalid_argument&) {
        }
      }
    CHECK(c.tip().hash() == ref.tip().hash());
  }
}

static void test_structure() {
  Task t = test_task();
  Block g = make_genesis(t, h_of(0x33));
  Block c = child_of(g, 0x40);
  CHECK(!check_structure(c, g, t));

  Block bad = c;
  bad.header.prev_hash = h_of(0);
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.header.w_start = h_of(0x99);
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.ckpt_hashes.pop_back();
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.ckpt_hashes[2] = h_of(0x77);  // checkpoint swapped, root left unchanged
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.header.K = 200;
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.header.height = 5;
  CHECK(check_structure(bad, g, t).has_value());

  // The "hash" field from the network is not trusted.
  json j = block_to_json(c);
  j["hash"] = to_hex(h_of(0));
  CHECK(block_from_json(j).hash() == c.hash());
}

static void test_chain() {
  Task t = test_task();
  Chain chain(make_genesis(t, h_of(0x33)));
  Block a1 = child_of(chain.genesis(), 0x40);
  Block b1 = child_of(chain.genesis(), 0x50);
  CHECK(chain.add(a1));
  chain.add(b1);  // equal height: the smaller hash wins, regardless of order
  Hash smaller = std::min(a1.hash(), b1.hash());
  CHECK(chain.tip().hash() == smaller);
  Chain reversed(make_genesis(t, h_of(0x33)));
  reversed.add(b1);
  reversed.add(a1);
  CHECK(reversed.tip().hash() == smaller);
  Block b2 = child_of(b1, 0x60);
  CHECK(chain.add(b2));  // branch b got longer: reorg
  CHECK(chain.tip().hash() == b2.hash());
  auto best = chain.best_chain();
  CHECK(best.size() == 3 && best[1].hash() == b1.hash());
  CHECK(chain.best_from(2).size() == 1);

  auto path = (std::filesystem::temp_directory_path() / "pouw_chain_test.json").string();
  chain.save(path);
  Chain restored(make_genesis(t, h_of(0x33)));
  CHECK(restored.load(path) == 2);
  CHECK(restored.tip().hash() == b2.hash());
  Chain other(make_genesis(t, h_of(0x34)));  // different genesis: the file is ignored
  CHECK(other.load(path) == 0);
  std::filesystem::remove(path);
}

static void test_event_bus() {
  auto path = (std::filesystem::temp_directory_path() / "pouw_events_test.jsonl").string();
  std::filesystem::remove(path);
  EventBus bus(path);
  CHECK(bus.wait_after(0, 10).empty());  // timeout with no events
  bus.publish("tip", {{"height", 1}});
  bus.publish("tip", {{"height", 2}});
  auto evs = bus.wait_after(1, 10);
  CHECK(evs.size() == 1 && evs[0].at("seq") == 2 && evs[0].at("data").at("height") == 2);
  std::thread late([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    bus.publish("mined", json::object());
  });
  auto woke = bus.wait_after(2, 2000);  // woken by an event, not by the timeout
  late.join();
  CHECK(woke.size() == 1 && woke[0].at("type") == "mined");
  std::ifstream in(path);
  int lines = 0;
  for (std::string l; std::getline(in, l);) ++lines;
  CHECK(lines == 3);
  std::filesystem::remove(path);
}

int main(int argc, char** argv) {
  test_hex();
  test_golden(argc > 1 ? argv[1] : "docs/golden.json");
  test_vectors(argc > 2 ? argv[2] : "docs/vectors.json");
  test_structure();
  test_chain();
  test_chain_order_independent();
  test_event_bus();
  if (g_failed) {
    std::cerr << g_failed << " check(s) failed\n";
    return 1;
  }
  std::cout << "all node tests passed\n";
  return 0;
}
