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

// Що API може спитати у вузла. Окремі функції замість посилання на Node: API не знає про його нутрощі.
struct ApiHandlers {
  std::function<json()> status;
  std::function<json(uint64_t from_height)> chain_from;
  std::function<std::optional<json>(const std::string& hash_hex)> block;
  std::function<json(const json& request)> admin;  // {"mode": "honest"|"lazy"} або {"paused": bool}
  EventBus* events = nullptr;
};

// HTTP API вузла лише на 127.0.0.1:
//   GET  /status                 стан вузла
//   GET  /chain?from=H           найкращий ланцюг від висоти H (за замовчуванням 0)
//   GET  /block/<hash>           блок за хешем
//   GET  /events?since=SEQ       потік подій (Server-Sent Events); також заголовок Last-Event-ID
//   POST /admin                  {"mode": "lazy"} / {"mode": "honest"} / {"paused": true}
class Api {
public:
  Api(uint16_t port, ApiHandlers handlers);
  ~Api();

  void start();  // окремий потік
  void stop();

private:
  uint16_t port_;
  ApiHandlers h_;
  std::unique_ptr<httplib::Server> server_;
  std::thread thread_;
};

}  // namespace pouw
