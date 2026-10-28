#include "block.hpp"

#include <stdexcept>

namespace pouw {

namespace {

template <typename T>
void put_le(uint8_t*& p, T v) {
  auto u = static_cast<std::make_unsigned_t<T>>(v);
  for (size_t i = 0; i < sizeof(T); ++i) *p++ = static_cast<uint8_t>(u >> (8 * i));
}

void put_hash(uint8_t*& p, const Hash& h) {
  std::copy(h.begin(), h.end(), p);
  p += h.size();
}

}  // namespace

std::vector<uint8_t> Header::serialize() const {
  std::vector<uint8_t> out(version >= 2 ? kHeaderSizeV2 : kHeaderSizeV1);
  uint8_t* p = out.data();
  put_le(p, version);
  put_le(p, height);
  put_hash(p, prev_hash);
  put_hash(p, task_id);
  put_le(p, round);
  put_hash(p, miner_id);
  put_le(p, timestamp);
  put_hash(p, w_start);
  put_hash(p, w_end);
  put_hash(p, ckpt_root);
  put_le(p, K);
  put_le(p, k);
  put_le(p, metric_micro);
  if (version >= 2) put_hash(p, tx_root);
  if (p != out.data() + out.size()) throw std::logic_error("header layout mismatch");
  return out;
}

Hash Header::hash() const { return sha256(serialize()); }

Task task_from_json(const json& j, const std::string& raw_text) {
  Task t;
  // Without an explicit task_id, the task id is the hash of the task file itself.
  t.task_id = j.contains("task_id") ? hash_from_hex(j.at("task_id").get<std::string>())
                                    : sha256(raw_text);
  t.task_seed = j.at("task_seed").get<uint64_t>();
  t.K = j.at("K").get<uint32_t>();
  t.k = j.at("k").get<uint32_t>();
  t.batch_size = j.value("batch_size", t.batch_size);
  t.lr = j.value("lr", t.lr);
  t.dataset = j.value("dataset", t.dataset);
  if (t.k == 0 || t.K % t.k != 0) throw std::invalid_argument("task: K must be a multiple of k");
  return t;
}

json header_to_json(const Header& h) {
  json j = {
      {"version", h.version},
      {"height", h.height},
      {"prev_hash", to_hex(h.prev_hash)},
      {"task_id", to_hex(h.task_id)},
      {"round", h.round},
      {"miner_id", to_hex(h.miner_id)},
      {"timestamp", h.timestamp},
      {"w_start", to_hex(h.w_start)},
      {"w_end", to_hex(h.w_end)},
      {"ckpt_root", to_hex(h.ckpt_root)},
      {"K", h.K},
      {"k", h.k},
      {"metric_micro", h.metric_micro},
  };
  if (h.version >= 2) j["tx_root"] = to_hex(h.tx_root);
  return j;
}

Header header_from_json(const json& j) {
  Header h;
  h.version = j.at("version").get<uint32_t>();
  h.height = j.at("height").get<uint64_t>();
  h.prev_hash = hash_from_hex(j.at("prev_hash").get<std::string>());
  h.task_id = hash_from_hex(j.at("task_id").get<std::string>());
  h.round = j.at("round").get<uint64_t>();
  h.miner_id = hash_from_hex(j.at("miner_id").get<std::string>());
  h.timestamp = j.at("timestamp").get<uint64_t>();
  h.w_start = hash_from_hex(j.at("w_start").get<std::string>());
  h.w_end = hash_from_hex(j.at("w_end").get<std::string>());
  h.ckpt_root = hash_from_hex(j.at("ckpt_root").get<std::string>());
  h.K = j.at("K").get<uint32_t>();
  h.k = j.at("k").get<uint32_t>();
  h.metric_micro = j.at("metric_micro").get<int64_t>();
  if (h.version >= 2) h.tx_root = hash_from_hex(j.at("tx_root").get<std::string>());
  return h;
}

json block_to_json(const Block& b) {
  json ckpts = json::array();
  for (const auto& c : b.ckpt_hashes) ckpts.push_back(to_hex(c));
  json txs = json::array();
  for (const auto& tx : b.txs) txs.push_back(tx_to_json(tx));
  return {{"hash", to_hex(b.hash())}, {"header", header_to_json(b.header)}, {"ckpt_hashes", ckpts},
          {"txs", txs}, {"sig", to_hex(b.sig)}};
}

// The "hash" field from the network is ignored: the hash is always recomputed from the header.
Block block_from_json(const json& j) {
  Block b;
  b.header = header_from_json(j.at("header"));
  for (const auto& c : j.at("ckpt_hashes")) b.ckpt_hashes.push_back(hash_from_hex(c.get<std::string>()));
  if (j.contains("txs")) {
    if (j.at("txs").size() > kMaxTxsPerBlock) throw std::invalid_argument("too many transfers");
    for (const auto& t : j.at("txs")) b.txs.push_back(tx_from_json(t));
  }
  if (j.contains("sig")) b.sig = signature_from_hex(j.at("sig").get<std::string>());
  return b;
}

Hash ckpt_root_of(const std::vector<Hash>& ckpt_hashes) {
  std::vector<uint8_t> buf;
  buf.reserve(ckpt_hashes.size() * 32);
  for (const auto& h : ckpt_hashes) buf.insert(buf.end(), h.begin(), h.end());
  return sha256(buf);
}

Block make_genesis(const Task& task, const Hash& w0) {
  Block g;
  g.header.task_id = task.task_id;
  g.header.w_start = w0;
  g.header.w_end = w0;
  g.header.K = task.K;
  g.header.k = task.k;
  g.ckpt_hashes = {w0};
  g.header.ckpt_root = ckpt_root_of(g.ckpt_hashes);
  g.header.tx_root = tx_root_of(g.txs);
  return g;
}

std::optional<std::string> check_structure(const Block& b, const Block& parent, const Task& task) {
  const Header& h = b.header;
  if (h.version != kBlockVersion) return "unsupported version";
  if (h.task_id != task.task_id) return "wrong task_id";
  if (h.prev_hash != parent.hash()) return "prev_hash does not link to parent";
  if (h.height != parent.header.height + 1) return "height is not parent+1";
  if (h.round != h.height) return "round must equal height";
  if (h.w_start != parent.header.w_end) return "w_start differs from parent w_end";  // D1
  if (h.K != task.K || h.k != task.k) return "K/k differ from task";                 // D3
  if (b.ckpt_hashes.size() != h.K / h.k + 1) return "wrong number of checkpoints";
  if (b.ckpt_hashes.front() != h.w_start) return "C0 differs from w_start";
  if (b.ckpt_hashes.back() != h.w_end) return "C_m differs from w_end";
  if (ckpt_root_of(b.ckpt_hashes) != h.ckpt_root) return "ckpt_root mismatch";
  if (h.metric_micro < 0 || h.metric_micro > 1'000'000) return "metric out of range";
  if (tx_root_of(b.txs) != h.tx_root) return "tx_root mismatch";
  if (!verify(h.miner_id, b.hash(), b.sig)) return "bad miner signature";
  return std::nullopt;
}

}  // namespace pouw
