#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <string>

#include <monocypher-ed25519.h>

#include "block.hpp"
#include "chain.hpp"
#include "mempool.hpp"
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

static KeyPair key_of(uint8_t byte) { return KeyPair::from_seed(h_of(byte)); }

// Honest-looking child block: m+1 checkpoints, C₀ = parent's w_end, signed by the miner key `salt`.
static Block child_of(const Block& parent, uint8_t salt, std::vector<Tx> txs = {}) {
  Block b;
  KeyPair key = key_of(salt);
  Header& h = b.header;
  h.height = parent.header.height + 1;
  h.prev_hash = parent.hash();
  h.task_id = parent.header.task_id;
  h.round = h.height;
  h.miner_id = key.public_key();
  h.w_start = parent.header.w_end;
  h.K = 100;
  h.k = 25;
  b.ckpt_hashes.push_back(h.w_start);
  for (uint8_t i = 1; i <= 4; ++i) b.ckpt_hashes.push_back(h_of(static_cast<uint8_t>(salt + i)));
  h.w_end = b.ckpt_hashes.back();
  h.ckpt_root = ckpt_root_of(b.ckpt_hashes);
  h.metric_micro = 900000;
  b.txs = std::move(txs);
  h.tx_root = tx_root_of(b.txs);
  b.sign(key);
  return b;
}

static Tx transfer(uint8_t from, const PublicKey& to, uint64_t amount, uint64_t fee, uint64_t nonce,
                   const Hash& chain) {
  Tx tx;
  tx.chain = chain;
  tx.to = to;
  tx.amount = amount;
  tx.fee = fee;
  tx.nonce = nonce;
  tx.sign(key_of(from));
  return tx;
}

static void test_ed25519() {
  // RFC 8032, section 7.1, test 1 (empty message): Monocypher's Ed25519 is the standard one.
  auto seed = from_hex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
  std::array<uint8_t, 64> secret;
  std::array<uint8_t, 32> pub;
  crypto_ed25519_key_pair(secret.data(), pub.data(), seed.data());
  CHECK(to_hex(pub) == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
  std::array<uint8_t, 64> sig;
  crypto_ed25519_sign(sig.data(), secret.data(), nullptr, 0);
  CHECK(to_hex(sig) ==
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46b"
        "d25bf5f0595bbe24655141438e7a100b");

  KeyPair k = KeyPair::from_seed(hash_from_hex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"));
  CHECK(to_hex(k.public_key()) == to_hex(pub));
  Hash msg = sha256(std::string_view("block"));
  Signature s = k.sign(msg);
  CHECK(verify(k.public_key(), msg, s));
  CHECK(!verify(k.public_key(), sha256(std::string_view("other")), s));
  s[10] ^= 1;
  CHECK(!verify(k.public_key(), msg, s));

  auto path = (std::filesystem::temp_directory_path() / "pouw_key_test.key").string();
  std::filesystem::remove(path);
  KeyPair created = KeyPair::load_or_create(path);
  CHECK(KeyPair::load(path).public_key() == created.public_key());
  CHECK((std::filesystem::status(path).permissions() & std::filesystem::perms::group_read) ==
        std::filesystem::perms::none);
  bool refused = false;
  try {
    created.save(path);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  CHECK(refused);
  std::filesystem::remove(path);
}

static void test_ledger() {
  Task t = test_task();
  Chain chain(make_genesis(t, h_of(0x33)));
  const PublicKey a = key_of(0x40).public_key(), b = key_of(0x50).public_key(), c = key_of(0x60).public_key();
  Block b1 = child_of(chain.genesis(), 0x40);  // a mines: +50
  CHECK(chain.add(b1));
  CHECK(chain.tip_ledger().account(a).balance == kBlockReward);

  Block b2 = child_of(b1, 0x60, {transfer(0x40, b, 20, 1, 0, t.task_id)});  // c mines a's transfer
  CHECK(!check_structure(b2, b1, t));
  CHECK(chain.add(b2));
  const Ledger& l = chain.tip_ledger();
  CHECK(l.account(a).balance == 29 && l.account(a).nonce == 1);
  CHECK(l.account(b).balance == 20);
  CHECK(l.account(c).balance == kBlockReward + 1);
  CHECK(l.supply() == 2 * kBlockReward);
  // The ledger after b1 is untouched: forks keep their own balances.
  CHECK(chain.ledger(b1.hash()).account(b).balance == 0);

  Ledger scratch;
  auto rejects = [&](std::vector<Tx> txs) { return chain.try_apply(child_of(b2, 0x70, std::move(txs)), scratch); };
  CHECK(rejects({transfer(0x40, b, 1, 0, 0, t.task_id)}).has_value());    // nonce 0 already used
  CHECK(rejects({transfer(0x40, b, 30, 0, 1, t.task_id)}).has_value());   // only 29 left
  CHECK(rejects({transfer(0x40, b, 1, 0, 1, h_of(0x12))}).has_value());   // signed for another network
  CHECK(rejects({transfer(0x40, a, 1, 0, 1, t.task_id)}).has_value());    // to self
  Tx forged = transfer(0x40, b, 1, 0, 1, t.task_id);
  forged.amount = 9;  // changed after signing
  CHECK(rejects({forged}).has_value());
  CHECK(!rejects({transfer(0x40, b, 9, 0, 1, t.task_id), transfer(0x40, c, 20, 0, 2, t.task_id)}));
  CHECK(rejects({transfer(0x40, b, 9, 0, 1, t.task_id), transfer(0x40, c, 21, 0, 2, t.task_id)}).has_value());

  // A block with an invalid transfer never enters the chain.
  bool threw = false;
  try {
    chain.add(child_of(b2, 0x70, {transfer(0x50, a, 21, 0, 0, t.task_id)}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  // tx_root covers transfer signatures: a copy with a mangled signature has a different block hash, so a
  // peer cannot get the genuine block banned by relaying a broken copy of it first.
  Tx good = transfer(0x40, b, 1, 0, 1, t.task_id), mangled = good;
  mangled.sig[5] ^= 1;
  CHECK(mangled.id() == good.id() && tx_root_of({mangled}) != tx_root_of({good}));

  // Transfers round-trip through JSON with their signature intact.
  Tx tx = transfer(0x40, b, 5, 2, 7, t.task_id);
  Tx back = tx_from_json(tx_to_json(tx));
  CHECK(back.id() == tx.id() && back.signature_ok());
}

static void test_hex() {
  Hash h = h_of(0xab);
  CHECK(hash_from_hex(to_hex(h)) == h);
  CHECK(to_hex(sha256(std::string_view("abc"))).rfind("ba7816bf", 0) == 0);
}

static void test_base64() {
  // RFC 4648 test vectors, then a round trip over every byte value and length remainder.
  const std::pair<std::string, std::string> rfc[] = {
      {"", ""}, {"f", "Zg=="}, {"fo", "Zm8="}, {"foo", "Zm9v"}, {"foob", "Zm9vYg=="}, {"fooba", "Zm9vYmE="},
      {"foobar", "Zm9vYmFy"}};
  for (const auto& [plain, enc] : rfc) {
    std::vector<uint8_t> bytes(plain.begin(), plain.end());
    CHECK(to_base64(bytes) == enc);
    CHECK(from_base64(enc) == bytes);
  }
  std::vector<uint8_t> all;
  for (int i = 0; i < 256; ++i) all.push_back(static_cast<uint8_t>(i));
  for (size_t n : {size_t{254}, size_t{255}, size_t{256}}) {
    std::vector<uint8_t> part(all.begin(), all.begin() + static_cast<long>(n));
    CHECK(from_base64(to_base64(part)) == part);
  }
  for (const char* bad : {"Zm9", "Zm9v!A==", "Z=9v"}) {
    bool threw = false;
    try {
      from_base64(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
}

static void test_golden(const std::string& path) {
  std::ifstream in(path);
  CHECK(static_cast<bool>(in));
  if (!in) return;
  json g = json::parse(in);
  Header h = header_from_json(g.at("header"));
  CHECK(h.serialize().size() == kHeaderSizeV1);  // the thesis-era golden block is version 1
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

  bad = c;
  bad.sig[0] ^= 1;  // tampered signature
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.sign(key_of(0x41));  // signed by someone other than miner_id
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.txs.push_back(transfer(0x40, h_of(9), 1, 0, 0, t.task_id));  // transfer added, tx_root unchanged
  CHECK(check_structure(bad, g, t).has_value());

  bad = c;
  bad.header.version = 1;
  bad.sign(key_of(0x40));
  CHECK(check_structure(bad, g, t).has_value());

  // The "hash" field from the network is not trusted.
  json j = block_to_json(c);
  j["hash"] = to_hex(h_of(0));
  CHECK(block_from_json(j).hash() == c.hash());
  CHECK(!check_structure(block_from_json(block_to_json(c)), g, t));  // signature survives the round trip
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
  Chain other(make_genesis(t, h_of(0x34)));  // different genesis: refused, so the node does not overwrite it
  bool refused = false;
  try {
    other.load(path);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  CHECK(refused && other.height() == 0);
  std::filesystem::remove(path);
}

static void test_mempool() {
  Task t = test_task();
  Chain chain(make_genesis(t, h_of(0x33)));
  const PublicKey b = key_of(0x50).public_key();
  Block b1 = child_of(chain.genesis(), 0x40);  // a: 50 coins
  chain.add(b1);
  const Ledger& tip = chain.tip_ledger();
  Mempool mp;
  auto tx = [&](uint64_t amount, uint64_t nonce) { return transfer(0x40, b, amount, 0, nonce, t.task_id); };

  CHECK(mp.offer(tx(20, 0), tip, t.task_id).result == Mempool::Result::ready);
  CHECK(mp.offer(tx(20, 0), tip, t.task_id).reason == "already known");
  CHECK(mp.offer(tx(21, 0), tip, t.task_id).result == Mempool::Result::rejected);  // nonce 0 taken
  CHECK(mp.offer(tx(5, 2), tip, t.task_id).result == Mempool::Result::waiting);    // gap: nonce 1 missing
  CHECK(mp.next_nonce(key_of(0x40).public_key(), tip) == 1);
  CHECK(mp.offer(tx(20, 1), tip, t.task_id).result == Mempool::Result::ready);     // fills the gap...
  CHECK(mp.ready_count() == 3 && mp.waiting_count() == 0);                          // ...and pulls nonce 2 in
  CHECK(mp.offer(tx(20, 3), tip, t.task_id).result == Mempool::Result::waiting);   // 45 of 50 already spent
  Tx forged = tx(1, 4);
  forged.amount = 2;
  CHECK(mp.offer(forged, tip, t.task_id).result == Mempool::Result::rejected);     // bad signature
  CHECK(mp.offer(tx(1, 99), tip, t.task_id).result == Mempool::Result::rejected);  // too far ahead

  auto picked = mp.select(tip);
  CHECK(picked.size() == 3 && picked[0].nonce == 0 && picked[2].nonce == 2);

  // The first two are mined on top of b1 (by a, who earns 50 more): they leave, nonce 3 becomes affordable.
  Block b2 = child_of(b1, 0x40, {picked[0], picked[1]});
  chain.add(b2);
  auto relay = mp.retarget(chain.tip_ledger(), {});
  CHECK(mp.ready_count() == 2 && mp.waiting_count() == 0);
  CHECK(relay.size() == 1 && relay[0].nonce == 3);

  // Reorg: a longer branch from b1 without those transfers. They come back from the abandoned block.
  Block c2 = child_of(b1, 0x60), c3 = child_of(c2, 0x60);
  chain.add(c2);
  chain.add(c3);
  auto gone = chain.abandoned(b2.hash());
  CHECK(gone.size() == 1 && gone[0].hash() == b2.hash());
  mp.retarget(chain.tip_ledger(), gone[0].txs);
  // On the new branch a has 50 again: nonces 0 and 1 (40) fit, nonce 2 (45 total) fits, nonce 3 does not.
  CHECK(mp.ready_count() == 3 && mp.waiting_count() == 1);
  CHECK(mp.next_nonce(key_of(0x40).public_key(), chain.tip_ledger()) == 3);
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
  test_base64();
  test_ed25519();
  test_ledger();
  test_mempool();
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
