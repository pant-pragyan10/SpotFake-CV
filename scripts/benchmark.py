from __future__ import annotations

import argparse
from pathlib import Path

from configs.config import ExperimentConfig, LabConfig
from src.experiments import ExperimentRunner, ExperimentRunnerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the feature evaluation laboratory.')
    parser.add_argument('--dataset-root', type=Path, default=Path('data/raw'), help='Root directory containing real/ and screen/.')
    parser.add_argument('--output-root', type=Path, default=Path('outputs'), help='Directory to write figures, reports, benchmarks, and csv files.')
    parser.add_argument('--latency-only', action='store_true', help='Only run latency benchmarking.')
    parser.add_argument('--visualization-only', action='store_true', help='Only run visualizations.')
    parser.add_argument('--statistics-only', action='store_true', help='Only run summary statistics.')
    args = parser.parse_args()

    lab = LabConfig(
        latency_only=args.latency_only,
        visualization_only=args.visualization_only,
        statistics_only=args.statistics_only,
    )
    runner = ExperimentRunner(
        ExperimentRunnerConfig(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            lab=lab,
            experiment=ExperimentConfig(),
        )
    )
    runner.run()


if __name__ == "__main__":
    main()
