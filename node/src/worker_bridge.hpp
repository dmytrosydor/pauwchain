#pragma once

#include <sys/types.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <string>

#include "block.hpp"

namespace pouw {

// Python worker invocation (docs/contract.md §3). Two modes:
//  - spawn: task.json → `python worker/worker.py task.json` → result.json, a separate process per call;
//  - server: one persistent `python worker/server.py` on a Unix socket (§3.5), started in the constructor.
//    Python, torch and the dataset load once, so a call costs only the work itself.
class WorkerBridge {
public:
  WorkerBridge(std::string python, std::string script, std::string work_dir);
  WorkerBridge(std::string python, std::string server_script, std::string work_dir,
               std::string socket_path, const std::string& preload_dataset);
  ~WorkerBridge();
  WorkerBridge(const WorkerBridge&) = delete;
  WorkerBridge& operator=(const WorkerBridge&) = delete;

  // Returns the worker result. nullopt if the call was cancelled via cancel.
  // Worker errors arrive as {"ok": false, "error": ...}.
  std::optional<json> run(json task, const std::atomic<bool>* cancel = nullptr);

private:
  std::optional<json> run_spawn(json task, const std::atomic<bool>* cancel);
  std::optional<json> run_socket(const json& task, const std::atomic<bool>* cancel);

  std::string python_;
  std::string script_;
  std::string work_dir_;
  std::string socket_path_;  // empty means spawn mode
  pid_t server_pid_ = 0;
  std::atomic<uint64_t> counter_{0};
};

}  // namespace pouw
