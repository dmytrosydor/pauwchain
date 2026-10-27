#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <string>

#include "block.hpp"

namespace pouw {

// Виклик Python-воркера: task.json → `python worker/worker.py task.json` → result.json
// (docs/contract.md §3). Кожен виклик — окремий процес.
class WorkerBridge {
public:
  WorkerBridge(std::string python, std::string script, std::string work_dir);

  // Повертає вміст result.json. nullopt — якщо виклик скасовано через cancel.
  // Помилки воркера приходять як {"ok": false, "error": ...}.
  std::optional<json> run(json task, const std::atomic<bool>* cancel = nullptr);

private:
  std::string python_;
  std::string script_;
  std::string work_dir_;
  std::atomic<uint64_t> counter_{0};
};

}  // namespace pouw
