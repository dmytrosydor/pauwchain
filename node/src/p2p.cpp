#include "p2p.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace pouw {

namespace {

constexpr uint32_t kMaxFrame = 64u << 20;

bool write_all(int fd, const void* data, size_t len) {
  auto* p = static_cast<const uint8_t*>(data);
  while (len > 0) {
    ssize_t n = ::write(fd, p, len);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool read_all(int fd, void* data, size_t len) {
  auto* p = static_cast<uint8_t*>(data);
  while (len > 0) {
    ssize_t n = ::read(fd, p, len);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

void tune_socket(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}

}  // namespace

P2P::P2P(uint16_t port, OnConnect on_connect, OnMessage on_message)
    : port_(port), on_connect_(std::move(on_connect)), on_message_(std::move(on_message)) {}

P2P::~P2P() { stop(); }

void P2P::start() {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) throw std::runtime_error("socket() failed");
  int one = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port_);
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    throw std::runtime_error("bind() failed on port " + std::to_string(port_) + ": " +
                             std::strerror(errno));
  }
  if (::listen(listen_fd_, 16) < 0) throw std::runtime_error("listen() failed");
  running_ = true;
  std::thread([this] { accept_loop(); }).detach();
}

void P2P::accept_loop() {
  while (running_) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      if (!running_) return;
      continue;
    }
    std::thread([this, fd] { serve(fd); }).detach();
  }
}

void P2P::connect_to(const std::string& host, uint16_t port) {
  std::thread([this, host, port] {
    while (running_) {
      addrinfo hints{};
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_STREAM;
      addrinfo* res = nullptr;
      if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) == 0) {
        int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd >= 0 && ::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
          freeaddrinfo(res);
          serve(fd);  // повертається після розриву, далі перепідключення
        } else {
          if (fd >= 0) ::close(fd);
          freeaddrinfo(res);
        }
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }).detach();
}

int P2P::add_peer(int fd) {
  std::lock_guard lock(mu_);
  int id = next_id_++;
  peers_[id] = std::make_shared<Peer>(fd);
  return id;
}

void P2P::remove_peer(int id) {
  std::lock_guard lock(mu_);
  peers_.erase(id);
}

void P2P::serve(int fd) {
  tune_socket(fd);
  int id = add_peer(fd);
  on_connect_(id);
  while (running_) {
    uint32_t be_len = 0;
    if (!read_all(fd, &be_len, 4)) break;
    uint32_t len = ntohl(be_len);
    if (len > kMaxFrame) break;
    std::string payload(len, '\0');
    if (!read_all(fd, payload.data(), len)) break;
    try {
      on_message_(id, json::parse(payload));
    } catch (const std::exception& e) {
      std::cerr << "p2p: bad message from peer " << id << ": " << e.what() << '\n';
    }
  }
  remove_peer(id);
  ::close(fd);
}

void P2P::send(int peer, const json& msg) {
  std::shared_ptr<Peer> p;
  {
    std::lock_guard lock(mu_);
    auto it = peers_.find(peer);
    if (it == peers_.end()) return;
    p = it->second;
  }
  std::string payload = msg.dump();
  uint32_t be_len = htonl(static_cast<uint32_t>(payload.size()));
  std::lock_guard lock(p->send_mu);
  if (!write_all(p->fd, &be_len, 4) || !write_all(p->fd, payload.data(), payload.size())) {
    ::shutdown(p->fd, SHUT_RDWR);  // читач помітить розрив і прибере пір
  }
}

void P2P::broadcast(const json& msg, int except) {
  std::vector<int> ids;
  {
    std::lock_guard lock(mu_);
    for (const auto& [id, _] : peers_) {
      if (id != except) ids.push_back(id);
    }
  }
  for (int id : ids) send(id, msg);
}

size_t P2P::peer_count() const {
  std::lock_guard lock(mu_);
  return peers_.size();
}

void P2P::stop() {
  if (!running_.exchange(false)) return;
  if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR), ::close(listen_fd_);
  std::lock_guard lock(mu_);
  for (const auto& [_, p] : peers_) ::shutdown(p->fd, SHUT_RDWR);
}

}  // namespace pouw
