"""Run manifest: which code, environment and parameters produced the results."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_manifest(out_dir: str, config: dict) -> dict:
    cfg = json.dumps(config, sort_keys=True, default=str)
    manifest = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "mps_available": torch.backends.mps.is_available(),
        "argv": sys.argv,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_sha256": hashlib.sha256(cfg.encode()).hexdigest(),
        "config": config,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
