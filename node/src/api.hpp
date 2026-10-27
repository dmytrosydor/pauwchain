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
  EventBus* events = nullptr;
};

// Node HTTP API, on 127.0.0.1 only:
//   GET  /status                 node status
//   GET  /chain?from=H           best chain from height H (default 0)
//   GET  /block/<hash>           block by hash
//   GET  /events?since=SEQ       event stream (Server-Sent Events); also the Last-Event-ID header
//   POST /admin                  {"mode": "lazy"} / {"mode": "honest"} / {"paused": true}
class Api {
public:
  Api(uint16_t port, ApiHandlers handlers);
  ~Api();

  void start();  // separate thread
  void stop();

private:
  uint16_t port_;
  ApiHandlers h_;
  std::unique_ptr<httplib::Server> server_;
  std::thread thread_;
};

}  // namespace pouw
