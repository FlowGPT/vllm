#!/usr/bin/env bash
# Pin flashinfer 0.6.8.post1 (avoids v0.23 default 0.6.12 tactic=-1 regression).
set -euo pipefail

source="${1:?FLASHINFER_SOURCE (pip|local)}"
py_site="${2:?PY_SITE}"
version="${3:?FLASHINFER_VERSION}"
cuda_tag="${4:?FLASHINFER_CUDA}"

rm -rf \
    "${py_site}/flashinfer" \
    "${py_site}/flashinfer_cubin" \
    "${py_site}/flashinfer_jit_cache" \
    "${py_site}"/flashinfer_python-* \
    "${py_site}"/flashinfer_cubin-* \
    "${py_site}"/flashinfer_jit_cache-*

case "${source}" in
  pip)
    uv pip install --system \
        "flashinfer-python==${version}" \
        "flashinfer-cubin==${version}" \
        "flashinfer-jit-cache==${version}" \
        --extra-index-url "https://flashinfer.ai/whl/${cuda_tag}"
    ;;
  local)
    pkg_root="/tmp/fi068-packages"
    if [[ ! -d "${pkg_root}/flashinfer" ]]; then
        echo "missing ${pkg_root}/flashinfer (run extract-flashinfer-packages.sh)" >&2
        exit 1
    fi
    cp -a "${pkg_root}/flashinfer" "${py_site}/"
    cp -a "${pkg_root}/flashinfer_cubin" "${py_site}/"
    cp -a "${pkg_root}/flashinfer_jit_cache" "${py_site}/"
    cp -a "${pkg_root}/flashinfer_python-${version}.dist-info" "${py_site}/"
    cp -a "${pkg_root}/flashinfer_cubin-${version}.dist-info" "${py_site}/"
    cp -a "${pkg_root}/flashinfer_jit_cache-${version}+${cuda_tag}.dist-info" "${py_site}/"
    ;;
  *)
    echo "unknown FLASHINFER_SOURCE=${source}" >&2
    exit 1
    ;;
esac
