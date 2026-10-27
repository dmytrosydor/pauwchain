#include "api.hpp"

#include <httplib.h>

namespace pouw {

namespace {

void reply(httplib::Response& res, const json& body, int status = 200) {
  res.status = status;
  res.set_header("Access-Control-Allow-Origin", "*");
  res.set_content(body.dump(), "application/json");
}

std::string sse_frame(const json& ev) {
  return "id: " + std::to_string(ev.at("seq").get<uint64_t>()) + "\nevent: " +
         ev.at("type").get<std::string>() + "\ndata: " + ev.dump() + "\n\n";
}

}  // namespace

Api::Api(uint16_t port, ApiHandlers handlers)
    : port_(port), h_(std::move(handlers)), server_(std::make_unique<httplib::Server>()) {
  auto& s = *server_;

  s.Get("/status", [this](const httplib::Request&, httplib::Response& res) { reply(res, h_.status()); });

  s.Get("/chain", [this](const httplib::Request& req, httplib::Response& res) {
    uint64_t from = 0;
    if (req.has_param("from")) from = std::stoull(req.get_param_value("from"));
    reply(res, h_.chain_from(from));
  });

  s.Get(R"(/block/([0-9a-f]{64}))", [this](const httplib::Request& req, httplib::Response& res) {
    if (auto b = h_.block(req.matches[1])) {
      reply(res, *b);
    } else {
      reply(res, {{"error", "unknown block"}}, 404);
    }
  });

  s.Post("/admin", [this](const httplib::Request& req, httplib::Response& res) {
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
      reply(res, {{"error", "expected a JSON object"}}, 400);
      return;
    }
    reply(res, h_.admin(body));
  });

  s.Get("/events", [this](const httplib::Request& req, httplib::Response& res) {
    uint64_t since = 0;
    if (req.has_param("since")) {
      since = std::stoull(req.get_param_value("since"));
    } else if (req.has_header("Last-Event-ID")) {
      since = std::stoull(req.get_header_value("Last-Event-ID"));
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Access-Control-Allow-Origin", "*");
    auto last = std::make_shared<uint64_t>(since);
    res.set_chunked_content_provider("text/event-stream", [this, last](size_t, httplib::DataSink& sink) {
      auto evs = h_.events->wait_after(*last, 15000);
      if (evs.empty()) {
        static const std::string keepalive = ": keepalive\n\n";
        return sink.write(keepalive.data(), keepalive.size());
      }
      for (const auto& ev : evs) {
        std::string frame = sse_frame(ev);
        if (!sink.write(frame.data(), frame.size())) return false;
        *last = ev.at("seq").get<uint64_t>();
      }
      return true;
    });
  });

  s.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string what = "internal error";
    try {
      if (ep) std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      what = e.what();
    }
    reply(res, {{"error", what}}, 500);
  });
}

Api::~Api() { stop(); }

void Api::start() {
  if (!server_->bind_to_port("127.0.0.1", port_)) {
    throw std::runtime_error("API: cannot bind 127.0.0.1:" + std::to_string(port_));
  }
  thread_ = std::thread([this] { server_->listen_after_bind(); });
}

void Api::stop() {
  if (server_) server_->stop();
  if (thread_.joinable()) thread_.join();
}

}  // namespace pouw
