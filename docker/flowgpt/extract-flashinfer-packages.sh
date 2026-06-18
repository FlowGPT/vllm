#!/usr/bin/env bash
# Offline fallback: extract flashinfer 0.6.8.post1 from kaon-0.22-v1.2 into docker/flowgpt/fi068-packages/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/docker/flowgpt/fi068-packages"
IMAGE="${1:-ssadds/vllm:kaon-0.22-v1.2}"
PY_SITE="/usr/local/lib/python3.12/dist-packages"

echo "Extracting flashinfer packages from ${IMAGE} -> ${OUT}"
mkdir -p "${OUT}"

copy_pkg() {
    local name="$1"
    echo "  ${name}..."
    docker run --rm --entrypoint bash "${IMAGE}" -c \
        "cd '${PY_SITE}' && tar cf - '${name}'" | tar xf - -C "${OUT}"
}

copy_pkg flashinfer
copy_pkg flashinfer_cubin
copy_pkg flashinfer_jit_cache
copy_pkg flashinfer_python-0.6.8.post1.dist-info
copy_pkg flashinfer_cubin-0.6.8.post1.dist-info
copy_pkg flashinfer_jit_cache-0.6.8.post1+cu130.dist-info

du -sh "${OUT}"
echo "Done. Rebuild with: FLASHINFER_SOURCE=local docker build ..."
