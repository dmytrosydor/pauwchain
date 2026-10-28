#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "block.hpp"

namespace pouw {

// TCP message exchange. Frame: 4-byte length (big-endian, network order) + UTF-8 JSON.
// One thread per connection. Outgoing connections reconnect on their own.
class P2P {
public:
  using OnMessage = std::function<void(int peer, const json& msg)>;
  using OnConnect = std::function<void(int peer)>;

  P2P(std::string listen_addr, uint16_t port, OnConnect on_connect, OnMessage on_message);
  ~P2P();

  void start();
  void connect_to(const std::string& host, uint16_t port);
  void send(int peer, const json& msg);
  void broadcast(const json& msg, int except = -1);
  size_t peer_count() const;
  std::vector<int> peer_ids() const;
  void stop();

private:
  struct Peer {
    int fd;
    std::mutex send_mu;
  };

  void accept_loop();
  void serve(int fd);  // reads frames while the connection is alive
  int add_peer(int fd);
  void remove_peer(int id);

  std::string listen_addr_;
  uint16_t port_;
  OnConnect on_connect_;
  OnMessage on_message_;
  int listen_fd_ = -1;
  std::atomic<bool> running_{false};
  mutable std::mutex mu_;
  std::map<int, std::shared_ptr<Peer>> peers_;
  int next_id_ = 1;
};

}  // namespace pouw
