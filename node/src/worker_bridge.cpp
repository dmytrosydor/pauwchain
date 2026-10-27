#include "worker_bridge.hpp"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

extern char** environ;

namespace pouw {

WorkerBridge::WorkerBridge(std::string python, std::string script, std::string work_dir)
    : python_(std::move(python)), script_(std::move(script)), work_dir_(std::move(work_dir)) {
  std::filesystem::create_directories(work_dir_);
}

std::optional<json> WorkerBridge::run(json task, const std::atomic<bool>* cancel) {
  uint64_t n = counter_++;
  std::string op = task.value("op", "task");
  std::string task_path = work_dir_ + "/" + op + "-" + std::to_string(n) + ".json";
  std::string out_path = work_dir_ + "/" + op + "-" + std::to_string(n) + ".result.json";
  std::string log_path = work_dir_ + "/worker.log";
  task["out"] = out_path;
  std::filesystem::remove(out_path);
  std::ofstream(task_path) << task.dump(2) << '\n';

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log_path.c_str(),
                                   O_WRONLY | O_CREAT | O_APPEND, 0644);
  posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);

  std::vector<char*> argv = {python_.data(), script_.data(), task_path.data(), nullptr};
  pid_t pid = 0;
  int rc = posix_spawn(&pid, python_.c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (rc != 0) return json{{"ok", false}, {"error", "posix_spawn failed: " + std::to_string(rc)}};

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
