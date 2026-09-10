#!/usr/bin/env bash
# Thin overlay on vllm/vllm-openai:v0.29.0 — copies only vllm/*.py changed vs v0.29.0.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$ROOT" rev-parse --show-toplevel)"

BASE_TAG="${BASE_TAG:-v0.29.0}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.29.0}"
IMAGE="${IMAGE:-local/vllm:flowgpt-0.29.0-metric-length}"
TAG="${TAG:-$IMAGE}"

python3 "$ROOT/gen_overlay.py" --base-tag "$BASE_TAG" --base-image "$BASE_IMAGE" --out-dir "$ROOT"

echo "=== docker build $IMAGE ==="
docker build -f "$ROOT/Dockerfile.gen" -t "$IMAGE" "$ROOT"

if [[ "$TAG" != "$IMAGE" ]]; then
  docker tag "$IMAGE" "$TAG"
fi

PATCH_REV="$(git -C "$REPO" rev-parse --short HEAD)"
echo "=== verify $IMAGE @ $PATCH_REV ==="
docker run --rm --entrypoint python3 \
  -v "$ROOT/verify_in_image.py:/tmp/verify_in_image.py:ro" \
  "$IMAGE" /tmp/verify_in_image.py

echo "=== patch files ==="
cat "$ROOT/overlay.manifest"
echo "BUILT $TAG @ $PATCH_REV"
