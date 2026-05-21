#!/usr/bin/env bash
# Sync the current directory to the SSH mirror (same credentials as scp).
# Audio/video-style files (mp4, gif, mp3, etc.) are excluded from rsync.
# Default: charlie@www.ultrapp.online:/home/charlie/project/<dirname>/
#
# Requires: rsync on both ends (install: apt install rsync / brew install rsync).
# Pure scp cannot mirror cleanly (no --delete, slow); rsync over ssh is the usual approach.
#
# Usage:
#   ./sync_mirror_ultrapp.sh
#   ./sync_mirror_ultrapp.sh --dry-run
#   REMOTE_HOST=other.host ./sync_mirror_ultrapp.sh
#   SYNC_ROOT=/path/to/repo ./sync_mirror_ultrapp.sh

set -euo pipefail

ROOT="${SYNC_ROOT:-$(pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
PROJECT_NAME="$(basename "$ROOT")"

REMOTE_USER="${REMOTE_USER:-charlie}"
REMOTE_HOST="${REMOTE_HOST:-www.ultrapp.online}"
REMOTE_BASE="${REMOTE_BASE:-/home/charlie/project}"
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_BASE}/${PROJECT_NAME}}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    *)
      echo "Unknown option: $arg (only --dry-run / -n supported)" >&2
      exit 1
      ;;
  esac
done

RSYNC=(rsync -avz)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC+=(--dry-run)
fi

# Mirror: drop remote files not present locally (careful if you store extra files only on server).
RSYNC+=(--delete)

RSYNC+=(
  --human-readable
  --exclude ".venv/"
  --exclude "__pycache__/"
  --exclude "*.py[cod]"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude ".ruff_cache/"
  --exclude "*.egg-info/"
  --exclude ".DS_Store"
  --exclude "dist/*.tar.gz"
  # Audio / video (do not mirror large media)
  --exclude "*.mp4"
  --exclude "*.webm"
  --exclude "*.avi"
  --exclude "*.mov"
  --exclude "*.mkv"
  --exclude "*.m4v"
  --exclude "*.wmv"
  --exclude "*.flv"
  --exclude "*.mpg"
  --exclude "*.mpeg"
  --exclude "*.3gp"
  --exclude "*.ogv"
  --exclude "*.mp3"
  --exclude "*.wav"
  --exclude "*.flac"
  --exclude "*.aac"
  --exclude "*.m4a"
  --exclude "*.ogg"
  --exclude "*.opus"
  --exclude "*.wma"
  --exclude "*.gif"
)

echo "Sync: $ROOT/ -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
"${RSYNC[@]}" "$ROOT/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
