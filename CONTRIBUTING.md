# Contributing

pouwchain is a bachelor's thesis project. Issues and pull requests are welcome, and the easiest way in is a new attack for the [arena](arena/README.md). For a bigger change, open an issue first so we can agree on the approach.

## Setup

```sh
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
cmake -S . -B build && cmake --build build -j
```

## Layout

- `node/` — C++: block, chain, validation, P2P over TCP, HTTP API.
- `worker/` — Python: training, verification, persistent worker server.
- `sim/` — Python: simulator, attacks, mechanisms, experiments, plots.
- `arena/` — attack strategies for the arena.
- `tui/` — pouwtop, the network monitor.
- `docs/` — thesis materials, in Ukrainian.

## Before you open a pull request

CI runs ruff and pytest on Linux, CTest on Linux and macOS, and CTest under ASan/UBSan. Locally:

```sh
.venv/bin/ruff check .
.venv/bin/python -m pytest -q worker/tests sim/tests tui/tests
ctest --test-dir build
```

## Changing the contract

Python and C++ must agree byte for byte. `docs/contract.md` defines the contract and `docs/vectors.json` holds the vectors both sides test against. If you change the contract, update it and regenerate the vectors:

```sh
python scripts/gen_vectors.py
```

Only do this when the contract really changes. If the vectors fail otherwise, the code is what's wrong.

## Style

Match the surrounding code. C++ is formatted with the repository's `.clang-format`, and Python is checked with `ruff.toml`. Commits follow [Conventional Commits](https://www.conventionalcommits.org).
