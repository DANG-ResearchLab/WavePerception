#!/bin/bash
#SBATCH --job-name=lowto_pamsvit_fullLSF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=mrigpu
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/mridata/mahamm2/swh_estimation/experiments/28_May_lowto_experiments_v1/logs/lowto_pamsvit_fullLSF_%j.out
#SBATCH --error=/mnt/mridata/mahamm2/swh_estimation/experiments/28_May_lowto_experiments_v1/logs/lowto_pamsvit_fullLSF_%j.err

set -e

# ========================================================================
# PATHS
# ========================================================================
PROJECT_ROOT="/mnt/mridata/mahamm2/swh_estimation/experiments/28_May_lowto_experiments_v1"
CONFIG_PATH="$PROJECT_ROOT/configs/lowto_pamsvit_fullLSF.yaml"
MAIN_SCRIPT="$PROJECT_ROOT/main.py"

# ========================================================================
# CONDA ENV
# ========================================================================
source /home/mahamm2/miniconda3/etc/profile.d/conda.sh
conda activate dl_wave

# ========================================================================
# DEBUG INFO
# ========================================================================
echo "=========================================="
echo "LOWTO PAMS-ViT FULL (LSF ENABLED) JOB START"
echo "=========================================="
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "Python:       $(which python)"
echo "Project root: $PROJECT_ROOT"
echo "Config:       $CONFIG_PATH"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$PROJECT_ROOT/logs"

# ========================================================================
# GPU CHECK
# ========================================================================
echo "=========================================="
echo "GPU INFO"
echo "=========================================="
nvidia-smi

# ========================================================================
# FILE CHECK
# ========================================================================
for f in "$CONFIG_PATH" "$MAIN_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

echo "=========================================="
echo "DATASET CHECK"
echo "=========================================="
DATASET_DIR="$PROJECT_ROOT/../../dataset"
SPLITS_CSV="$DATASET_DIR/splits.csv"
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: dataset directory not found: $DATASET_DIR"
    exit 1
fi
if [ ! -f "$SPLITS_CSV" ]; then
    echo "ERROR: splits.csv not found: $SPLITS_CSV"
    exit 1
fi
echo "Dataset OK: $DATASET_DIR"
ls -lh "$SPLITS_CSV"

# ========================================================================
# RUN
# ========================================================================
echo "=========================================="
echo "START LOWTO PAMS-ViT FULL (LSF ENABLED)"
echo "=========================================="

python -u "$MAIN_SCRIPT" --config "$CONFIG_PATH"

echo "=========================================="
echo "LOWTO PAMS-ViT FULL (LSF ENABLED) DONE"
echo "Date: $(date)"
echo "=========================================="
