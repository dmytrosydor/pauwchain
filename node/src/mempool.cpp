#include "mempool.hpp"

#include <algorithm>

namespace pouw {

uint64_t Mempool::spent(const PublicKey& from) const {
  uint64_t total = 0;
  if (auto it = ready_.find(from); it != ready_.end()) {
    for (const auto& [_, tx] : it->second) total += tx.amount + tx.fee;  // ≤ the sender's balance: no overflow
  }
  return total;
}

bool Mempool::add_waiting(const Tx& tx) {
  Queue& q = waiting_[tx.from];
  if (q.count(tx.nonce)) return q.at(tx.nonce).id() == tx.id();
  // Full lists turn new transfers away instead of evicting old ones: spam from fresh keys cannot push out
  // transfers that are already here.
  if (q.size() >= kMaxWaitingPerSender || waiting_count_ >= kMaxWaiting) {
    if (q.empty()) waiting_.erase(tx.from);
    return false;
  }
  q.emplace(tx.nonce, tx);
  ++waiting_count_;
  return true;
}

void Mempool::promote(const PublicKey& from, const Ledger& tip, std::vector<Tx>* promoted) {
  auto wit = waiting_.find(from);
  if (wit == waiting_.end()) return;
  const Account& acc = tip.account(from);
  Queue& ready = ready_[from];
  uint64_t budget = acc.balance - spent(from);
  for (auto it = wit->second.begin(); it != wit->second.end();) {
    const Tx& tx = it->second;
    if (tx.nonce != acc.nonce + ready.size() || budget < tx.amount + tx.fee || ready_count_ >= kMaxReady) break;
    budget -= tx.amount + tx.fee;
    if (promoted) promoted->push_back(tx);
    ready.emplace(tx.nonce, tx);
    ++ready_count_;
    --waiting_count_;
    it = wit->second.erase(it);
  }
  if (wit->second.empty()) waiting_.erase(wit);
  if (ready.empty()) ready_.erase(from);
}

Mempool::Outcome Mempool::offer(const Tx& tx, const Ledger& tip, const Hash& chain) {
  const Account& acc = tip.account(tx.from);
  for (const auto* lists : {&ready_, &waiting_}) {
    if (auto it = lists->find(tx.from); it != lists->end()) {
      if (auto t = it->second.find(tx.nonce); t != it->second.end()) {
        return {Result::rejected, t->second.id() == tx.id() ? "already known"
                                                            : "another transfer with this nonce is pending"};
      }
    }
  }
  if (tx.nonce < acc.nonce) return {Result::rejected, "nonce already used"};
  if (tx.nonce > acc.nonce + kMaxNonceAhead) return {Result::rejected, "nonce too far ahead"};
  if (auto err = Ledger::check_tx(tx, chain)) return {Result::rejected, *err};

  const size_t queued = ready_.count(tx.from) ? ready_.at(tx.from).size() : 0;
  const bool next = tx.nonce == acc.nonce + queued;
  const bool funded = acc.balance - spent(tx.from) >= tx.amount + tx.fee;
  if (next && funded && ready_count_ < kMaxReady) {
    ready_[tx.from].emplace(tx.nonce, tx);
    ++ready_count_;
    promote(tx.from, tip, nullptr);  // later nonces may have been waiting for this one
    return {Result::ready, ""};
  }
  std::string why = !next ? "an earlier transfer of this sender is missing" : !funded ? "insufficient balance"
                                                                                       : "mempool full";
  if (!add_waiting(tx)) return {Result::rejected, why + "; waiting list full"};
  return {Result::waiting, why};
}

std::vector<Tx> Mempool::retarget(const Ledger& tip, const std::vector<Tx>& returned) {
  std::map<PublicKey, Queue> all;
  std::map<Hash, bool> was_ready;  // id → it was ready before
  for (const auto* lists : {&ready_, &waiting_}) {
    for (const auto& [from, q] : *lists) {
      for (const auto& [nonce, tx] : q) {
        all[from].emplace(nonce, tx);
        was_ready[tx.id()] = lists == &ready_;
      }
    }
  }
  // A transfer from an abandoned block was validated there, so its signature is fine; a pending one with
  // the same nonce stays (it was here first).
  for (const Tx& tx : returned) all[tx.from].emplace(tx.nonce, tx);

  ready_.clear();
  waiting_.clear();
  ready_count_ = waiting_count_ = 0;
  std::vector<Tx> newly_ready;
  for (auto& [from, q] : all) {
    const Account& acc = tip.account(from);
    uint64_t budget = acc.balance;
    uint64_t expect = acc.nonce;
    for (auto& [nonce, tx] : q) {
      if (nonce < acc.nonce) continue;  // mined on the new branch
      if (nonce == expect && budget >= tx.amount + tx.fee && ready_count_ < kMaxReady) {
        budget -= tx.amount + tx.fee;
        ++expect;
        auto it = was_ready.find(tx.id());
        if (it == was_ready.end() || !it->second) newly_ready.push_back(tx);
        ready_[from].emplace(nonce, tx);
        ++ready_count_;
      } else if (nonce <= acc.nonce + kMaxNonceAhead) {
        add_waiting(tx);
      }
    }
  }
  return newly_ready;
}

std::vector<Tx> Mempool::select(const Ledger& parent) const {
  std::vector<Tx> out;
  for (const auto& [from, q] : ready_) {
    Account acc = parent.account(from);
    for (const auto& [nonce, tx] : q) {
      if (out.size() >= kMaxTxsPerBlock) return out;
      if (nonce != acc.nonce || acc.balance < tx.amount + tx.fee) break;
      acc.balance -= tx.amount + tx.fee;
      ++acc.nonce;
      out.push_back(tx);
    }
  }
  return out;
}

uint64_t Mempool::next_nonce(const PublicKey& key, const Ledger& tip) const {
  auto it = ready_.find(key);
  return tip.account(key).nonce + (it == ready_.end() ? 0 : it->second.size());
}

std::vector<Tx> Mempool::ready() const {
  std::vector<Tx> out;
  for (const auto& [_, q] : ready_) {
    for (const auto& [__, tx] : q) out.push_back(tx);
  }
  return out;
}

std::vector<Tx> Mempool::waiting() const {
  std::vector<Tx> out;
  for (const auto& [_, q] : waiting_) {
    for (const auto& [__, tx] : q) out.push_back(tx);
  }
  return out;
}

}  // namespace pouw
