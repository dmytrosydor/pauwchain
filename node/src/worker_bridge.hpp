#pragma once

#include <sys/types.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <string>

#include "block.hpp"

namespace pouw {

// Виклик Python-воркера (docs/contract.md §3). Два режими:
//  - spawn: task.json → `python worker/worker.py task.json` → result.json, окремий процес на виклик;
//  - server: один постійний `python worker/server.py` на Unix socket (§3.5), запускається в конструкторі.
//    Python, torch і набір даних завантажуються один раз, тож виклик коштує лише саму роботу.
class WorkerBridge {
public:
  WorkerBridge(std::string python, std::string script, std::string work_dir);
  WorkerBridge(std::string python, std::string server_script, std::string work_dir,
               std::string socket_path, const std::string& preload_dataset);
  ~WorkerBridge();
  WorkerBridge(const WorkerBridge&) = delete;
  WorkerBridge& operator=(const WorkerBridge&) = delete;

  // Повертає результат воркера. nullopt — якщо виклик скасовано через cancel.
  // Помилки воркера приходять як {"ok": false, "error": ...}.
  std::optional<json> run(json task, const std::atomic<bool>* cancel = nullptr);

private:
  std::optional<json> run_spawn(json task, const std::atomic<bool>* cancel);
  std::optional<json> run_socket(const json& task, const std::atomic<bool>* cancel);

  std::string python_;
  std::string script_;
  std::string work_dir_;
  std::string socket_path_;  // порожній — режим spawn
  pid_t server_pid_ = 0;
  std::atomic<uint64_t> counter_{0};
};

}  // namespace pouw
