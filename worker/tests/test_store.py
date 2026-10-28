"""Store import: a weights file from a peer enters the store only if it matches the requested hash."""
import os

import torch
from safetensors.torch import save_file

from worker.model import init_state
from worker.store import import_file, load_state, save_state


def test_import_accepts_matching_file(tmp_path):
    state = init_state(1)
    h = save_state(str(tmp_path / "src"), state)
    incoming = tmp_path / "in.tmp"
    os.replace(tmp_path / "src" / f"{h}.safetensors", incoming)

    assert import_file(str(tmp_path / "store"), str(incoming), h)
    assert not incoming.exists()
    loaded = load_state(str(tmp_path / "store"), h)
    assert all(torch.equal(loaded[k], state[k]) for k in state)


def test_import_rejects_wrong_or_broken_file(tmp_path):
    h = save_state(str(tmp_path / "src"), init_state(1))
    other = tmp_path / "other.tmp"
    save_file({k: v.contiguous() for k, v in init_state(2).items()}, str(other))
    broken = tmp_path / "broken.tmp"
    broken.write_bytes(b"not a safetensors file")

    for path in (other, broken):
        assert not import_file(str(tmp_path / "store"), str(path), h)
        assert not path.exists()
    assert not (tmp_path / "store" / f"{h}.safetensors").exists()
