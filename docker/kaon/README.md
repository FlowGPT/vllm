# kaon production image build assets

Reproduces `ssadds/vllm:kaon-0.22-v1.2` from branch `kaon-0.22-v1.2`.

## Quick build (flashinfer via pip)

```bash
git checkout kaon-0.22-v1.2
chmod +x docker/kaon/*.sh
./docker/kaon/build-kaon-0.22-v1.2.sh
```

## Bit-exact flashinfer layer (matches production COPY)

Production replaces flashinfer with `fi068-packages` (~11 GB). To reproduce that layer exactly:

```bash
./docker/kaon/extract-flashinfer-packages.sh ssadds/vllm:kaon-0.22-v1.2
./docker/kaon/build-kaon-0.22-v1.2.sh --flashinfer-source=local
```

`fi068-packages/` is gitignored; only the extract script is versioned.

## Image layout

| Layer | Source |
|---|---|
| Base | `vllm/vllm-openai:nightly-4721bb3aa` |
| Fork patches | 6 files from `kaon-0.22-v1.2` branch |
| flashinfer | `0.6.8.post1` (+ `cu130` jit-cache) |
| ENTRYPOINT | `[]` (K8s passes `vllm serve ...`) |
