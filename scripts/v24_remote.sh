#!/usr/bin/env bash
# Remote control for the BC -> v24 pipeline.
#
# Usage:
#   bash scripts/v24_remote.sh status        # procs + latest logs
#   bash scripts/v24_remote.sh sync          # rsync code to remote
#   bash scripts/v24_remote.sh collect       # parallel v20 self-play collection
#   bash scripts/v24_remote.sh collect-tail  # collection progress
#   bash scripts/v24_remote.sh train-bc      # train BC on the merged dataset
#   bash scripts/v24_remote.sh bc-tail       # BC training progress
#   bash scripts/v24_remote.sh h2h [N]       # export BC ckpt + replay vs v20 (N games)
#   bash scripts/v24_remote.sh start [nupd]  # start v24 PPO fine-tune
#   bash scripts/v24_remote.sh stop          # stop v24
#   bash scripts/v24_remote.sh tail          # v24 train log tail
#   bash scripts/v24_remote.sh eval          # v24 inline vs-v20 evals
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
RPY="/home/charlie/anaconda3/bin/python"

cmd="${1:-status}"
shift || true

_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== gpu ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && echo '=== procs ===' && ps aux | grep -E 'collect_dat[a]|train_b[c]|v24_bc_f[t]' | grep -v grep || echo '(none)'"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      --exclude='_archive/' \
      "$ROOT/" "$REMOTE:$RDIR/"
    _ssh "chmod +x '$RDIR/scripts/v24_bc_ft.sh' '$RDIR/scripts/v24_remote.sh' '$RDIR/scripts/bc_collect_parallel.sh'"
    echo "synced"
    ;;
  collect)
    NPROC="${1:-8}"; GAMES="${2:-50}"
    _ssh "cd '$RDIR' && (nohup bash scripts/bc_collect_parallel.sh '$NPROC' '$GAMES' data/bc_v20_self_400g.npz > logs/bc_collect_main.log 2>&1 &) && sleep 3 && tail -5 logs/bc_collect_main.log"
    ;;
  collect-tail)
    _ssh "cd '$RDIR' && tail -3 logs/bc_collect_main.log && echo '--- per-shard last line ---' && for f in logs/bc_collect/shard_*.log; do echo \"\$f: \$(grep 'game ' \"\$f\" | tail -1)\"; done"
    ;;
  train-bc)
    EPOCHS="${1:-8}"
    _ssh "cd '$RDIR' && (nohup env PYTHONPATH='$RDIR' '$RPY' -m orbit_wars_rl.bc.train_bc --data data/bc_v20_self_400g.npz --epochs '$EPOCHS' --batch-size 256 --lr 3e-4 --out ckpt_bc_v20/ckpt_final.pkl > logs/bc_train.log 2>&1 &) && sleep 3 && tail -5 logs/bc_train.log"
    ;;
  bc-tail)
    _ssh "tail -15 '$RDIR/logs/bc_train.log'"
    ;;
  h2h)
    N="${1:-20}"
    _ssh "cd '$RDIR' && NUM_GAMES='$N' JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHON='$RPY' nohup bash scripts/quick_replay.sh ckpt_bc_v20/ckpt_final.pkl bc_v20_clone > logs/bc_h2h.log 2>&1 & sleep 3; echo 'h2h started, check: bash scripts/v24_remote.sh h2h-tail'"
    ;;
  h2h-tail)
    _ssh "tail -8 '$RDIR/logs/bc_h2h.log'; echo '---'; cat '$RDIR/logs/replay_analyze/bc_v20_clone_vs_v20.summary.txt' 2>/dev/null | head -8 || echo '(summary not ready)'"
    ;;
  start)
    NUPD="${1:-4000}"
    _ssh "cd '$RDIR' && (nohup bash scripts/v24_bc_ft.sh '$NUPD' > logs/v24_launch_\$(date +%Y%m%d_%H%M%S).log 2>&1 &) && sleep 3 && ps aux | grep 'v24_bc_f[t]' | grep -v grep && ls -t logs/v24_bc_ft_*/train.log 2>/dev/null | head -1"
    ;;
  stop)
    _ssh "pkill -f 'multi_action_v24_bc_f[t].yaml' 2>/dev/null || true; pkill -f 'v24_bc_f[t].sh' 2>/dev/null || true; sleep 2; ps aux | grep 'v24_bc_f[t]' | grep python || echo 'v24 stopped'"
    ;;
  tail)
    _ssh "tail -10 \$(ls -t '$RDIR'/logs/v24_bc_ft_*/train.log 2>/dev/null | head -1)"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v24_bc_ft_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -15"
    ;;
  *)
    echo "usage: $0 {status|sync|collect|collect-tail|train-bc|bc-tail|h2h|h2h-tail|start|stop|tail|eval}" >&2
    exit 2
    ;;
esac
