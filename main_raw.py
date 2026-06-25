# =============================================================================
# main.py — Entry point
# =============================================================================
"""
Usage:
    python main.py --config configs/lowto_cnn.yaml

The config drives everything. To run a different experiment, pass a
different config file — no code changes needed.
"""

import argparse
import logging
from pathlib import Path

from src.utils import (set_seed, get_device, load_config, create_run_dir,
                        save_config_snapshot, setup_logging,
                        collect_environment_info)
from src.experiments.lowto import run_lowto


def parse_args():
    p = argparse.ArgumentParser(description="LOWTO experiment runner")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file.")
    p.add_argument("--resume_run", type=str, default=None,
                   help="Existing run directory to resume "
                        "(skip creating a new timestamped folder).")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # ---- Run directory ----
    if args.resume_run is not None:
        run_dir = Path(args.resume_run)
        if not run_dir.exists():
            raise FileNotFoundError(f"resume_run not found: {run_dir}")
    else:
        run_dir = create_run_dir(
            base_dir=config["output"]["runs_root"],
            experiment_name=config["experiment"]["name"],
            timestamp_format=config["output"].get(
                "timestamp_format", "%Y%m%d_%H%M%S"),
        )

    # ---- Logging ----
    setup_logging(log_file=run_dir / "run_log.txt", level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config:        {args.config}")
    logger.info(f"Environment:   {collect_environment_info()}")

    # ---- Seed + device ----
    set_seed(int(config["experiment"]["seed"]))
    device = get_device(config.get("device", "auto"))
    logger.info(f"Device:        {device}")

    # ---- Snapshot config to run folder ----
    save_config_snapshot(config, run_dir)

    # ---- Dispatch experiment ----
    mode = config["split"]["mode"].lower()
    if mode == "lowto":
        run_lowto(config, device, run_dir)
    else:
        raise NotImplementedError(f"split.mode={mode!r} not supported "
                                  f"(only 'lowto' for now)")

    logger.info("Done.")


if __name__ == "__main__":
    main()