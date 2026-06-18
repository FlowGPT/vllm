#!/usr/bin/env bash
# Build ssadds/vllm:kaon-0.22-v1.2 from branch kaon-0.22-v1.2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

IMAGE_TAG="${IMAGE_TAG:-ssadds/vllm:kaon-0.22-v1.2}"
VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-vllm/vllm-openai:nightly-4721bb3aa43078167eb893a9ebf9e50565030c1c}"
FLASHINFER_SOURCE="${FLASHINFER_SOURCE:-pip}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flashinfer-source=*) FLASHINFER_SOURCE="${1#*=}"; shift ;;
        --image-tag=*) IMAGE_TAG="${1#*=}"; shift ;;
        --base-image=*) VLLM_BASE_IMAGE="${1#*=}"; shift ;;
        -h|--help)
            cat <<'EOF'
Usage: docker/kaon/build-kaon-0.22-v1.2.sh [options]

Options:
  --flashinfer-source=pip|local   pip (default) or local fi068-packages
  --image-tag=TAG                 output tag (default: ssadds/vllm:kaon-0.22-v1.2)
  --base-image=IMAGE              upstream base (default: nightly-4721bb3aa)
EOF
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

branch="$(git branch --show-current)"
if [[ "${branch}" != "kaon-0.22-v1.2" ]]; then
    echo "warning: not on kaon-0.22-v1.2 (current: ${branch})" >&2
fi

echo "Building ${IMAGE_TAG} (flashinfer=${FLASHINFER_SOURCE}, base=${VLLM_BASE_IMAGE})"
docker build \
    -f docker/Dockerfile.kaon-0.22-v1.2 \
    --build-arg VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE}" \
    --build-arg FLASHINFER_SOURCE="${FLASHINFER_SOURCE}" \
    -t "${IMAGE_TAG}" \
    .

echo "Built ${IMAGE_TAG}"
echo "Verify: docker run --rm --entrypoint python3 ${IMAGE_TAG} -c \"import flashinfer; print(flashinfer.__version__)\""
