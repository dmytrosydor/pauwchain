#!/usr/bin/env bash
# Local network: 3 honest nodes (+ 1 lazy attacker) on MNIST.
# Usage: scripts/demo.sh            — up to height 5, with a lazy node
#        HEIGHT=8 LAZY=0 scripts/demo.sh
#        HEIGHT=0 scripts/demo.sh — network runs until Ctrl-C (for pouwtop)
# Variables: HEIGHT (5), LAZY (1), TIMEOUT seconds (300), PYTHON (.venv/bin/python), K (200), k (20),
#            WORKER_MODE (server: persistent worker; spawn: a new Python process per call),
#            STORE (separate: each node keeps its own weights and fetches missing checkpoints from peers;
#                   shared: one directory for all nodes, as before).
# API of node i: http://127.0.0.1:$((9200 + i)); watch with: .venv/bin/python -m tui
set -euo pipefail
cd "$(dirname "$0")/.."

HEIGHT=${HEIGHT:-5}
LAZY=${LAZY:-1}
TIMEOUT=${TIMEOUT:-300}
PYTHON=${PYTHON:-.venv/bin/python}
K=${K:-200}
k=${k:-20}
WORKER_MODE=${WORKER_MODE:-server}
STORE=${STORE:-separate}
DEMO=build/demo
BASE_PORT=9100
API_PORT=9200

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j --target node >/dev/null

rm -rf "$DEMO"
mkdir -p "$DEMO"
cat >"$DEMO/task.json" <<EOF
{"task_seed": 42, "K": $K, "k": $k, "batch_size": 64, "lr": 0.05, "dataset": "mnist"}
EOF
if [ "$STORE" = shared ]; then
  # Genesis weights once, so nodes don't write the same file concurrently.
  echo '{"op": "genesis", "task_seed": 42, "store_dir": "'"$DEMO"'/store"}' >"$DEMO/genesis.json"
  "$PYTHON" worker/worker.py "$DEMO/genesis.json" >/dev/null
fi
store_of() { [ "$STORE" = shared ] && echo "$DEMO/store" || echo "$DEMO/node-$1/store"; }

ids=(1 2 3)
[ "$LAZY" = 1 ] && ids+=(4)
peers_of() {
  local me=$1 out=""
  for j in "${ids[@]}"; do
    [ "$j" = "$me" ] || out+="127.0.0.1:$((BASE_PORT + j)),"
  done
  echo "${out%,}"
}

pids=()
cleanup() { [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null || true; }
trap cleanup EXIT

for i in "${ids[@]}"; do
  extra=()
  [ "$i" = 4 ] && extra=(--lazy)
  build/node --id "$i" --port $((BASE_PORT + i)) --peers "$(peers_of "$i")" \
    --task "$DEMO/task.json" --store "$(store_of "$i")" --run-dir "$DEMO/node-$i" \
    --python "$PYTHON" --max-height "$HEIGHT" --api-port $((API_PORT + i)) --worker-mode "$WORKER_MODE" \
    ${extra[@]+"${extra[@]}"} >"$DEMO/node-$i.log" 2>&1 &
  pids+=($!)
done
echo "started nodes ${ids[*]} (K=$K, k=$k, target height $HEIGHT, lazy=$LAZY, worker=$WORKER_MODE, store=$STORE), logs in $DEMO/"
echo "API: $(for i in "${ids[@]}"; do printf 'http://127.0.0.1:%s ' $((API_PORT + i)); done)"

if [ "$HEIGHT" = 0 ]; then
  echo "running until Ctrl-C; in another terminal: .venv/bin/python -m tui"
  wait
  exit 0
fi

# Wait until all honest nodes reach the target height and converge on one tip.
converged() {
  "$PYTHON" - "$DEMO" "$HEIGHT" <<'EOF'
import json, sys
demo, target = sys.argv[1], int(sys.argv[2])
try:
    st = [json.load(open(f"{demo}/node-{i}/status.json")) for i in (1, 2, 3)]
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if all(s["height"] >= target for s in st) and len({s["tip"] for s in st}) == 1 else 1)
EOF
}
start=$(date +%s)
until converged; do
  if [ $(($(date +%s) - start)) -ge "$TIMEOUT" ]; then echo "timeout after ${TIMEOUT}s"; break; fi
  sleep 1
done
cleanup
trap - EXIT

echo
echo "== log excerpt (node 1) =="
grep -E 'MINED|ACCEPT|REJECT|lost race' "$DEMO/node-1.log" | head -20
echo
echo "== summary =="
"$PYTHON" - "$DEMO" "${ids[@]}" <<'EOF'
import json, sys
demo, ids = sys.argv[1], sys.argv[2:]
chains = {}
for i in ids:
    s = json.load(open(f"{demo}/node-{i}/status.json"))
    c = json.load(open(f"{demo}/node-{i}/chain.json"))
    chains[i] = [b["hash"] for b in c["blocks"]]
    print(f"node {i}{' (lazy)' if s['lazy'] else '       '}: height={s['height']} tip={s['tip'][:10]} "
          f"mined={s['mined']} accepted={s['accepted']} rejected={s['rejected']} lost_races={s['lost_races']} "
          f"blobs fetched/served={s['blobs_fetched']}/{s['blobs_served']} balance={s['balance']}")
honest = [chains[i] for i in ("1", "2", "3")]
common = min(len(c) for c in honest)
agree = 0
while agree < common and len({c[agree] for c in honest}) == 1:
    agree += 1
print(f"honest nodes agree on the first {agree} blocks (genesis + {agree - 1}); "
      f"tips equal: {len({c[-1] for c in honest}) == 1}")
lazy_id = json.load(open(f"{demo}/node-4/status.json"))["miner_id"] if "4" in ids else ""
lazy_rejects = sum(1 for i in ("1", "2", "3") for line in open(f"{demo}/node-{i}.log")
                   if "REJECT" in line and "miner=" + lazy_id[:10] in line)
if "4" in ids:
    lazy_in_chains = sum(1 for i in ("1", "2", "3")
                         for b in json.load(open(f"{demo}/node-{i}/chain.json"))["blocks"]
                         if b["header"]["miner_id"] == lazy_id)
    print(f"lazy blocks rejected by honest nodes: {lazy_rejects}; lazy blocks in honest chains: {lazy_in_chains}")
EOF
