#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

output_root=/efs-exp/egosteer/zhangtingrui/outputs
mkdir -p "$output_root" data
if [[ -L data/outputs ]]; then
    if [[ "$(readlink data/outputs)" != "$output_root" ]]; then
        echo "data/outputs already points elsewhere; keep it or update it explicitly before this run." >&2
        exit 1
    fi
elif [[ -e data/outputs ]]; then
    echo "data/outputs already exists; move its contents explicitly before creating the persistent-storage symlink." >&2
    exit 1
else
    ln -s "$output_root" data/outputs
fi

# Workspace scales LR from the actual process count, per-rank batch, and accumulation.
# Each rank has multiple loader workers; keep BLAS/OpenMP pools single-threaded.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
    train_torchrun.py --config-name=train_diffusion_transformer_hybrid_wds_workspace_stack_cup "$@"
