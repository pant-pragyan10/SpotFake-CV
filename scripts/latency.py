from __future__ import annotations

import argparse
from pathlib import Path

from configs.config import ExperimentConfig, LabConfig
from src.experiments import ExperimentRunner, ExperimentRunnerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description='Run latency benchmarking for the feature extractors.')
    parser.add_argument('--dataset-root', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    args = parser.parse_args()

    runner = ExperimentRunner(
        ExperimentRunnerConfig(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            lab=LabConfig(
                latency_only=True,
                run_statistics=False,
                run_visualizations=False,
                run_correlation=False,
                run_importance=False,
                run_ablation=False,
                run_reports=False,
            ),
            experiment=ExperimentConfig(),
        )
    )
    runner.run()


if __name__ == "__main__":
    main()
