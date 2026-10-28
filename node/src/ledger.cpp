#include "ledger.hpp"

#include <limits>
#include <stdexcept>

namespace pouw {

std::array<uint8_t, Tx::kSize> Tx::serialize() const {
  std::array<uint8_t, kSize> out{};
  uint8_t* p = out.data();
  for (char c : {'P', 'W', 'T', 'X'}) *p++ = static_cast<uint8_t>(c);
  for (const Hash* h : {&chain, &from, &to}) p = std::copy(h->begin(), h->end(), p);
  for (uint64_t v : {amount, fee, nonce}) {
    for (int i = 0; i < 8; ++i) *p++ = static_cast<uint8_t>(v >> (8 * i));
  }
  return out;
}

Hash Tx::full_hash() const {
  auto bytes = serialize();
  std::vector<uint8_t> buf(bytes.begin(), bytes.end());
  buf.insert(buf.end(), sig.begin(), sig.end());
  return sha256(buf);
}

void Tx::sign(const KeyPair& key) {
  from = key.public_key();
  sig = key.sign(id());
}

json tx_to_json(const Tx& tx) {
  return {{"id", to_hex(tx.id())},     {"chain", to_hex(tx.chain)}, {"from", to_hex(tx.from)},
          {"to", to_hex(tx.to)},       {"amount", tx.amount},       {"fee", tx.fee},
          {"nonce", tx.nonce},         {"sig", to_hex(tx.sig)}};
}

// The "id" field is ignored: it is always recomputed from the content.
Tx tx_from_json(const json& j) {
  Tx tx;
  tx.chain = hash_from_hex(j.at("chain").get<std::string>());
  tx.from = hash_from_hex(j.at("from").get<std::string>());
  tx.to = hash_from_hex(j.at("to").get<std::string>());
  tx.amount = j.at("amount").get<uint64_t>();
  tx.fee = j.at("fee").get<uint64_t>();
  tx.nonce = j.at("nonce").get<uint64_t>();
  tx.sig = signature_from_hex(j.at("sig").get<std::string>());
  return tx;
}

Hash tx_root_of(const std::vector<Tx>& txs) {
  std::vector<uint8_t> buf;
  buf.reserve(txs.size() * 32);
  for (const auto& tx : txs) {
    Hash h = tx.full_hash();
    buf.insert(buf.end(), h.begin(), h.end());
  }
  return sha256(buf);
}

const Account& Ledger::account(const PublicKey& key) const {
  static const Account kEmpty;
  auto it = accounts_.find(key);
  return it == accounts_.end() ? kEmpty : it->second;
}

std::optional<std::string> Ledger::check_tx(const Tx& tx, const Hash& chain) {
  if (tx.chain != chain) return "transfer is for another network";
  if (tx.from == tx.to) return "transfer to self";
  if (tx.amount == 0) return "zero amount";
  if (tx.amount > std::numeric_limits<uint64_t>::max() - tx.fee) return "amount + fee overflows";
  if (!tx.signature_ok()) return "bad signature";
  return std::nullopt;
}

std::optional<std::string> Ledger::apply(const std::vector<Tx>& txs, const PublicKey& miner, const Hash& chain) {
  if (txs.size() > kMaxTxsPerBlock) return "too many transfers";
  Ledger next = *this;
  uint64_t fees = 0;
  for (size_t i = 0; i < txs.size(); ++i) {
    const Tx& tx = txs[i];
    const std::string where = "transfer " + std::to_string(i) + ": ";
    if (auto err = check_tx(tx, chain)) return where + *err;
    Account& from = next.accounts_[tx.from];
    if (tx.nonce != from.nonce) {
      return where + "nonce " + std::to_string(tx.nonce) + ", expected " + std::to_string(from.nonce);
    }
    if (from.balance < tx.amount + tx.fee) return where + "insufficient balance";
    from.balance -= tx.amount + tx.fee;
    ++from.nonce;
    next.accounts_[tx.to].balance += tx.amount;  // cannot overflow: the total supply fits in uint64
    fees += tx.fee;
  }
  next.accounts_[miner].balance += kBlockReward + fees;
  next.supply_ += kBlockReward;
  *this = std::move(next);
  return std::nullopt;
}

}  // namespace pouw
