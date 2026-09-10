#!/usr/bin/env bash
# Append a marker comment, rebuild, and assert the image carries the new commit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$ROOT" rev-parse --show-toplevel)"
IMAGE="${IMAGE:-local/vllm:flowgpt-0.29.0-metric-length-test}"
MARKER="__flowgpt_metric_length_rebuild_test__"

cd "$REPO"
git checkout flowgpt-0.29.0-metric-length

marker_file="$REPO/vllm/config/observability.py"
saved_head="$(git rev-parse HEAD)"
printf '\n# %s\n' "$MARKER" >>"$marker_file"
git add "$marker_file"
git commit --no-verify -m "[Test] Rebuild loop marker for flowgpt-0.29.0-metric-length overlay"

IMAGE="$IMAGE" "$ROOT/build.sh"

REV="$(git rev-parse --short HEAD)"
docker run --rm --entrypoint grep "$IMAGE" -q "$MARKER" \
  /usr/local/lib/python3.12/dist-packages/vllm/config/observability.py

git reset --hard "$saved_head"
echo "REBUILD_OK $IMAGE @ $REV (test commit reset)"
