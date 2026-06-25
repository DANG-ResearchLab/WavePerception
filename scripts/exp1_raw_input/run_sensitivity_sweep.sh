#!/bin/bash
#SBATCH --job-name=sensitivity_pamsvit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=mrigpu
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/mridata/mahamm2/swh_estimation/experiments/28_May_lowto_experiments_v1/logs/sensitivity_%j.out
#SBATCH --error=/mnt/mridata/mahamm2/swh_estimation/experiments/28_May_lowto_experiments_v1/logs/sensitivity_%j.err

set -e

PROJECT_ROOT="/mnt/mridata/mahamm2/swh_estimation/experiments/28_May_lowto_experiments_v1"
CONFIG_PATH="$PROJECT_ROOT/configs/sensitivity_pamsvit_fullLSF.yaml"
ENTRY="$PROJECT_ROOT/main_sensitivity.py"
FOLD=5                    # interior fold, Hs=1.5 cm

source /home/mahamm2/miniconda3/etc/profile.d/conda.sh
conda activate dl_wave

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "SENSITIVITY ANALYSIS — PAMS-ViT (full LSF)"
echo "Fold: $FOLD (Hs=1.5 cm, interior)"
echo "Date: $(date)"
echo "=========================================="

# ------------------------------------------------------------------------
# Default baseline (to confirm reproducibility against full LOWTO result)
# ------------------------------------------------------------------------
echo "[1/9] Default baseline"
python -u "$ENTRY" --config "$CONFIG_PATH" --fold "$FOLD" \
    --run_name "default_baseline"

# ------------------------------------------------------------------------
# Sweep 1: embedding dimension (D)
# Default D = 64
# ------------------------------------------------------------------------
echo "[2/9] embed_dim = 32"
python -u "$ENTRY" --config "$CONFIG_PATH" --fold "$FOLD" \
    --embed_dim 32 --run_name "embed_dim_32"

echo "[3/9] embed_dim = 128"
python -u "$ENTRY" --config "$CONFIG_PATH" --fold "$FOLD" \
    --embed_dim 128 --run_name "embed_dim_128"

# ------------------------------------------------------------------------
# Sweep 2: transformer depth (n_blocks)
# Default = 4
# ------------------------------------------------------------------------
echo "[4/9] n_blocks = 2"
python -u "$ENTRY" --config "$CONFIG_PATH" --fold "$FOLD" \
    --n_blocks 2 --run_name "n_blocks_2"

echo "[5/9] n_blocks = 6"
python -u "$ENTRY" --config "$CONFIG_PATH" --fold "$FOLD" \
    --n_blocks 6 --run_name "n_blocks_6"

# ------------------------------------------------------------------------
# Sweep 3: learning rate
# Default = 1e-4
# ------------------------------------------------------------------------
echo "[6/9] lr = 1e-5"
python -u "$ENTRY" --config "$CONFIG_PATH" --fold "$FOLD" \
    --lr 1e-5 --run_name "lr_1e-5"

echo "[7/9] lr = 1e-3"
python -u "$ENTRY" --config "$CONFIG_PATH" --fold "$FOLD" \
    --lr 1e-3 --run_name "lr_1e-3"

echo "=========================================="
echo "SENSITIVITY SWEEP COMPLETE"
echo "Date: $(date)"
echo "=========================================="

# Aggregate results
python -u "$PROJECT_ROOT/scripts/aggregate_sensitivity.py" \
    --runs_dir "$PROJECT_ROOT/outputs/sensitivity"

echo "Aggregation complete. See sensitivity_summary.csv"
