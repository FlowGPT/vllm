#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

IMAGE_TAG="${IMAGE_TAG:-ssadds/vllm:flowgpt-0.23-kv-gemma4}"
VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-vllm/vllm-openai:v0.23.0}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-flowgpt-0.23-kv-gemma4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image-tag=*) IMAGE_TAG="${1#*=}"; shift ;;
        --base-image=*) VLLM_BASE_IMAGE="${1#*=}"; shift ;;
        -h|--help)
            cat <<'EOF'
Build flowgpt-0.23-kv-gemma4 image (v0.23.0 + KV prefix cache + Gemma4 ModelOpt MoE + FI 0.6.8).

Usage: docker/flowgpt/build-0.23-kv-gemma4.sh [--image-tag=TAG] [--base-image=IMAGE]

Env:
  IMAGE_TAG          default ssadds/vllm:flowgpt-0.23-kv-gemma4
  VLLM_BASE_IMAGE    default vllm/vllm-openai:v0.23.0
EOF
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

branch="$(git branch --show-current)"
if [[ "${branch}" != "${EXPECTED_BRANCH}" ]]; then
    echo "warning: not on ${EXPECTED_BRANCH} (current: ${branch})" >&2
fi

echo "Building ${IMAGE_TAG} (base=${VLLM_BASE_IMAGE})"
docker build \
    -f docker/Dockerfile.flowgpt-0.23-kv-gemma4 \
    --build-arg VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    .

echo "Built ${IMAGE_TAG}"
docker run --rm --entrypoint python3 "${IMAGE_TAG}" -c \
    "import flashinfer, vllm; print('flashinfer', flashinfer.__version__); print('vllm', vllm.__version__)"
