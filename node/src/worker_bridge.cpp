#include "worker_bridge.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

extern char** environ;

namespace pouw {

namespace {

constexpr uint32_t kMaxResult = 8u << 20;

pid_t spawn_logged(const std::vector<std::string>& args, const std::string& log_path) {
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log_path.c_str(),
                                   O_WRONLY | O_CREAT | O_APPEND, 0644);
  posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
  std::vector<std::string> copy = args;
  std::vector<char*> argv;
  for (auto& a : copy) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = 0;
  int rc = posix_spawn(&pid, copy[0].c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  return rc == 0 ? pid : -1;
}

int connect_unix(const std::string& path) {
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (path.size() >= sizeof(addr.sun_path)) throw std::invalid_argument("socket path too long: " + path);
  std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

bool write_all(int fd, const void* data, size_t len) {
  auto* p = static_cast<const uint8_t*>(data);
  while (len > 0) {
    ssize_t n = ::write(fd, p, len);
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
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

}  // namespace

WorkerBridge::WorkerBridge(std::string python, std::string script, std::string work_dir)
    : python_(std::move(python)), script_(std::move(script)), work_dir_(std::move(work_dir)) {
  std::filesystem::create_directories(work_dir_);
}

WorkerBridge::WorkerBridge(std::string python, std::string server_script, std::string work_dir,
                           std::string socket_path, const std::string& preload_dataset)
    : python_(std::move(python)),
      script_(std::move(server_script)),
      work_dir_(std::move(work_dir)),
      socket_path_(std::move(socket_path)) {
  std::filesystem::create_directories(work_dir_);
  std::filesystem::remove(socket_path_);
  std::vector<std::string> args = {python_, script_, "--socket", socket_path_};
  if (!preload_dataset.empty()) args.insert(args.end(), {"--preload", preload_dataset});
  std::string log_path = work_dir_ + "/server.log";
  server_pid_ = spawn_logged(args, log_path);
  if (server_pid_ < 0) throw std::runtime_error("cannot start worker server " + script_);

  // Чекаємо, доки сервер почне приймати з'єднання (імпорт torch і завантаження даних — секунди).
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  for (;;) {
    int fd = connect_unix(socket_path_);
    if (fd >= 0) {
      close(fd);
      break;
    }
    int status = 0;
    if (waitpid(server_pid_, &status, WNOHANG) == server_pid_) {
      server_pid_ = 0;
      throw std::runtime_error("worker server exited, see " + log_path);
    }
    if (std::chrono::steady_clock::now() > deadline) {
      throw std::runtime_error("worker server did not start in time, see " + log_path);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

WorkerBridge::~WorkerBridge() {
  if (server_pid_ > 0) {
    kill(server_pid_, SIGTERM);
    int status = 0;
    waitpid(server_pid_, &status, 0);
  }
}

std::optional<json> WorkerBridge::run(json task, const std::atomic<bool>* cancel) {
  if (!socket_path_.empty()) return run_socket(task, cancel);
  return run_spawn(std::move(task), cancel);
}

std::optional<json> WorkerBridge::run_socket(const json& task, const std::atomic<bool>* cancel) {
  int fd = connect_unix(socket_path_);
  if (fd < 0) return json{{"ok", false}, {"error", "cannot connect to worker server " + socket_path_}};
  std::string payload = task.dump();
  uint32_t len = htonl(static_cast<uint32_t>(payload.size()));
  if (!write_all(fd, &len, 4) || !write_all(fd, payload.data(), payload.size())) {
    close(fd);
    return json{{"ok", false}, {"error", "worker server write failed"}};
  }
  // Чекаємо відповідь; скасування — просто закрити з'єднання, сервер зупинить навчання.
  pollfd p{fd, POLLIN, 0};
  for (;;) {
    if (cancel && cancel->load()) {
      close(fd);
      return std::nullopt;
    }
    int r = poll(&p, 1, 50);
    if (r > 0) break;
    if (r < 0 && errno != EINTR) {
      close(fd);
      return json{{"ok", false}, {"error", "worker server poll failed"}};
    }
  }
  uint32_t n = 0;
  std::string body;
  bool ok = read_all(fd, &n, 4);
  n = ntohl(n);
  if (ok && n <= kMaxResult) {
    body.resize(n);
    ok = read_all(fd, body.data(), n);
  }
  close(fd);
  if (!ok || n > kMaxResult) return json{{"ok", false}, {"error", "worker server closed the connection"}};
  json res = json::parse(body, nullptr, false);
  if (res.is_discarded()) return json{{"ok", false}, {"error", "bad result json from worker server"}};
  return res;
}

std::optional<json> WorkerBridge::run_spawn(json task, const std::atomic<bool>* cancel) {
  uint64_t n = counter_++;
  std::string op = task.value("op", "task");
  std::string task_path = work_dir_ + "/" + op + "-" + std::to_string(n) + ".json";
  std::string out_path = work_dir_ + "/" + op + "-" + std::to_string(n) + ".result.json";
  std::string log_path = work_dir_ + "/worker.log";
  task["out"] = out_path;
  std::filesystem::remove(out_path);
  std::ofstream(task_path) << task.dump(2) << '\n';

  pid_t pid = spawn_logged({python_, script_, task_path}, log_path);
  if (pid < 0) return json{{"ok", false}, {"error", "posix_spawn failed"}};

  int status = 0;
  while (waitpid(pid, &status, WNOHANG) == 0) {
    if (cancel && cancel->load()) {
      kill(pid, SIGTERM);
      waitpid(pid, &status, 0);
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::ifstream in(out_path);
  if (!in) {
    return json{{"ok", false}, {"error", "worker produced no result, see " + log_path}};
  }
  try {
    return json::parse(in);
  } catch (const std::exception& e) {
    return json{{"ok", false}, {"error", std::string("bad result json: ") + e.what()}};
  }
}

}  // namespace pouw
