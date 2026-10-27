#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "block.hpp"
#include "chain.hpp"

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

// Чесний на вигляд дочірній блок: m+1 точок, C₀ = w_end батька.
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
  bad.ckpt_hashes[2] = h_of(0x77);  // точку підмінили, корінь лишився старим
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.header.K = 200;
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.header.height = 5;
  CHECK(check_structure(bad, g, t).has_value());

  // Поле "hash" з мережі не довіряється.
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
  chain.add(b1);  // рівна висота: перемагає менший хеш, незалежно від порядку
  Hash smaller = std::min(a1.hash(), b1.hash());
  CHECK(chain.tip().hash() == smaller);
  Chain reversed(make_genesis(t, h_of(0x33)));
  reversed.add(b1);
  reversed.add(a1);
  CHECK(reversed.tip().hash() == smaller);
  Block b2 = child_of(b1, 0x60);
  CHECK(chain.add(b2));  // гілка b стала довшою — реорганізація
  CHECK(chain.tip().hash() == b2.hash());
  auto best = chain.best_chain();
  CHECK(best.size() == 3 && best[1].hash() == b1.hash());
  CHECK(chain.best_from(2).size() == 1);

  auto path = (std::filesystem::temp_directory_path() / "pouw_chain_test.json").string();
  chain.save(path);
  Chain restored(make_genesis(t, h_of(0x33)));
  CHECK(restored.load(path) == 2);
  CHECK(restored.tip().hash() == b2.hash());
  Chain other(make_genesis(t, h_of(0x34)));  // інший генезис — файл ігнорується
  CHECK(other.load(path) == 0);
  std::filesystem::remove(path);
}

int main(int argc, char** argv) {
  test_hex();
  test_golden(argc > 1 ? argv[1] : "docs/golden.json");
  test_structure();
  test_chain();
  if (g_failed) {
    std::cerr << g_failed << " check(s) failed\n";
    return 1;
  }
  std::cout << "all node tests passed\n";
  return 0;
}
