// pouw: wallet command line. Keeps an Ed25519 key in a file and talks to a node over its HTTP API.
//
//   pouw new                     create a wallet (~/.pouw/wallet.key) and print its address
//   pouw address                 print the wallet's address
//   pouw balance [ADDRESS]       balance of the wallet or of any address
//   pouw send ADDRESS AMOUNT     sign a transfer and submit it (--fee N, default 0)
//   pouw accounts                every account on the chain, richest first
//   pouw status                  what the node sees
//
// Options: --wallet PATH (or $POUW_WALLET), --node URL (or $POUW_NODE, default http://127.0.0.1:9201).
#include <httplib.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "ledger.hpp"

using namespace pouw;

namespace {

struct Args {
  std::string command;
  std::vector<std::string> positional;
  std::map<std::string, std::string> options;

  std::string opt(const std::string& name, const char* env, const std::string& fallback) const {
    if (auto it = options.find(name); it != options.end()) return it->second;
    if (const char* v = env ? std::getenv(env) : nullptr; v && *v) return v;
    return fallback;
  }
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + s);
      a.options[s.substr(2)] = argv[++i];
    } else if (a.command.empty()) {
      a.command = s;
    } else {
      a.positional.push_back(s);
    }
  }
  return a;
}

std::string default_wallet() {
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.pouw/wallet.key";
}

// Accepts the full 64-hex public key; a "pw" prefix, as printed by `pouw address`, is optional.
PublicKey parse_address(std::string s) {
  if (s.rfind("pw", 0) == 0) s = s.substr(2);
  try {
    return hash_from_hex(s);
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("not an address (expected pw + 64 hex characters): " + s);
  }
}

std::string show(const PublicKey& k) { return "pw" + to_hex(k); }

KeyPair load_wallet(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("no wallet at " + path + "; create one with `pouw new` or pass --wallet PATH");
  }
  return KeyPair::load(path);
}

class Node {
public:
  explicit Node(const std::string& url) : url_(url), client_(url) {
    client_.set_connection_timeout(3);
    client_.set_read_timeout(10);
  }

  json get(const std::string& path) { return check(client_.Get(path), "GET " + path); }
  json post(const std::string& path, const json& body) {
    return check(client_.Post(path, body.dump(), "application/json"), "POST " + path, true);
  }

private:
  json check(const httplib::Result& r, const std::string& what, bool body_on_error = false) {
    if (!r) throw std::runtime_error("cannot reach the node at " + url_ + " (is the network running?)");
    json j = json::parse(r->body, nullptr, false);
    if (j.is_discarded()) throw std::runtime_error(what + ": the node sent something that is not JSON");
    if (r->status >= 400 && !body_on_error) {
      throw std::runtime_error(what + ": " + j.value("error", std::to_string(r->status)));
    }
    return j;
  }

  std::string url_;
  httplib::Client client_;
};

uint64_t parse_amount(const std::string& s, const char* what) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument(std::string(what) + " must be a whole number of coins: " + s);
  }
  if (s.size() > 19) throw std::invalid_argument(std::string(what) + " is too large: " + s);  // < 10^19 < 2^64
  return std::stoull(s);
}

int run(const Args& a) {
  const std::string wallet = a.opt("wallet", "POUW_WALLET", default_wallet());
  const std::string url = a.opt("node", "POUW_NODE", "http://127.0.0.1:9201");

  if (a.command == "new") {
    KeyPair k = KeyPair::generate();
    k.save(wallet);
    std::cout << "new wallet: " << wallet << "\naddress:    " << show(k.public_key())
              << "\n\nKeep the file secret: whoever has it can spend the coins.\n";
    return 0;
  }
  if (a.command == "address") {
    std::cout << show(load_wallet(wallet).public_key()) << '\n';
    return 0;
  }
  if (a.command == "balance") {
    PublicKey who = a.positional.empty() ? load_wallet(wallet).public_key() : parse_address(a.positional[0]);
    json acc = Node(url).get("/account/" + to_hex(who));
    std::cout << acc.at("balance").get<uint64_t>() << " coins  (" << show(who).substr(0, 14) << "…, block "
              << acc.at("height").get<uint64_t>() << ")\n";
    uint64_t pending = acc.at("next_nonce").get<uint64_t>() - acc.at("nonce").get<uint64_t>();
    if (pending) std::cout << pending << " outgoing transfer(s) waiting for a block\n";
    return 0;
  }
  if (a.command == "send") {
    if (a.positional.size() != 2) throw std::invalid_argument("usage: pouw send ADDRESS AMOUNT [--fee N]");
    Tx tx;
    tx.to = parse_address(a.positional[0]);  // input errors first, before touching the wallet or the network
    tx.amount = parse_amount(a.positional[1], "amount");
    tx.fee = parse_amount(a.opt("fee", nullptr, "0"), "fee");
    KeyPair k = load_wallet(wallet);
    Node node(url);
    tx.chain = hash_from_hex(node.get("/status").at("task_id").get<std::string>());
    tx.nonce = node.get("/account/" + to_hex(k.public_key())).at("next_nonce").get<uint64_t>();
    tx.sign(k);
    json r = node.post("/tx", tx_to_json(tx));
    if (!r.value("accepted", false)) {
      std::cerr << "rejected: " << r.value("error", std::string("?")) << '\n';
      return 1;
    }
    std::cout << "sent " << tx.amount << " coins to " << show(tx.to).substr(0, 14) << "… (fee " << tx.fee
              << ")\ntransfer " << r.at("id").get<std::string>().substr(0, 16) << "… ";
    if (r.contains("waiting")) {
      std::cout << "is on hold: " << r.at("waiting").get<std::string>() << "; the node retries it with every block\n";
    } else {
      std::cout << "is waiting for the next block\n";
    }
    return 0;
  }
  if (a.command == "accounts") {
    json r = Node(url).get("/accounts");
    std::cout << "block " << r.at("height") << ", " << r.at("supply") << " coins in existence\n";
    for (const auto& acc : r.at("accounts")) {
      std::cout << "  " << show(hash_from_hex(acc.at("address").get<std::string>())).substr(0, 14) << "…  "
                << acc.at("balance") << '\n';
    }
    return 0;
  }
  if (a.command == "status") {
    json s = Node(url).get("/status");
    std::cout << "node " << s.at("id").get<std::string>() << (s.at("lazy").get<bool>() ? " (cheating)" : "")
              << ": block " << s.at("height") << ", model accuracy " << s.at("tip_metric_micro").get<int64_t>() / 1e4
              << " %\naddress " << show(hash_from_hex(s.at("miner_id").get<std::string>())) << "\nbalance "
              << s.at("balance") << " coins, " << s.at("mempool") << " transfer(s) waiting\n";
    return 0;
  }
  std::cerr << "usage: pouw new | address | balance [ADDRESS] | send ADDRESS AMOUNT [--fee N] | accounts | status\n"
               "       options: --wallet PATH (default ~/.pouw/wallet.key), --node URL (default "
               "http://127.0.0.1:9201)\n";
  return a.command.empty() || a.command == "help" ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse(argc, argv));
  } catch (const std::exception& e) {
    std::cerr << "pouw: " << e.what() << '\n';
    return 1;
  }
}
