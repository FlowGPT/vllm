#!/usr/bin/env bash
# Extract flashinfer 0.6.8.post1 packages from a running kaon image into
# docker/kaon/fi068-packages/ for bit-exact local rebuilds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/docker/kaon/fi068-packages"
IMAGE="${1:-ssadds/vllm:kaon-0.22-v1.2}"
PY_SITE="/usr/local/lib/python3.12/dist-packages"

echo "Extracting flashinfer packages from ${IMAGE} -> ${OUT}"
mkdir -p "${OUT}"

copy_pkg() {
    local name="$1"
    docker run --rm --entrypoint bash "${IMAGE}" -c \
        "cd '${PY_SITE}' && tar cf - '${name}'" | tar xf - -C "${OUT}"
}

copy_pkg flashinfer
copy_pkg flashinfer_cubin
copy_pkg flashinfer_jit_cache
copy_pkg flashinfer_python-0.6.8.post1.dist-info
copy_pkg flashinfer_cubin-0.6.8.post1.dist-info
copy_pkg flashinfer_jit_cache-0.6.8.post1+cu130.dist-info

echo "Done. Rebuild with:"
echo "  docker/kaon/build-kaon-0.22-v1.2.sh --flashinfer-source=local"
