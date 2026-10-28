#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "block.hpp"
#include "events.hpp"

namespace httplib {
class Server;
}

namespace pouw {

// What the API can ask the node. Separate functions instead of a Node reference: the API knows nothing of its internals.
struct ApiHandlers {
  std::function<json()> status;
  std::function<json(uint64_t from_height)> chain_from;
  std::function<std::optional<json>(const std::string& hash_hex)> block;
  std::function<json(const json& request)> admin;  // {"mode": "honest"|"lazy"} or {"paused": bool}
  std::function<json(const json& tx)> submit_tx;    // {"accepted": bool, "id"?, "error"?}
  std::function<json(const std::string& key_hex)> account;
  std::function<json()> accounts;                  // every account at the tip, richest first
  std::function<json()> mempool;
  EventBus* events = nullptr;
};

// Node HTTP API, on --api-listen (127.0.0.1 by default; /admin has no authentication):
//   GET  /status                 node status
//   GET  /chain?from=H           best chain from height H (default 0)
//   GET  /block/<hash>           block by hash
//   GET  /events?since=SEQ       event stream (Server-Sent Events); also the Last-Event-ID header
//   POST /admin                  {"mode": "lazy"} / {"mode": "honest"} / {"paused": true}
//   POST /tx                     a signed transfer (docs/contract.md §5)
//   GET  /account/<key>          balance, nonce and the next free nonce of an account
//   GET  /accounts               all balances at the tip
//   GET  /mempool                transfers waiting for a block
class Api {
public:
  Api(std::string listen_addr, uint16_t port, ApiHandlers handlers);
  ~Api();

  void start();  // separate thread
  void stop();

private:
  std::string listen_addr_;
  uint16_t port_;
  ApiHandlers h_;
  std::unique_ptr<httplib::Server> server_;
  std::thread thread_;
};

}  // namespace pouw
