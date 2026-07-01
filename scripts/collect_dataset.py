from __future__ import annotations

import argparse
from pathlib import Path

from configs.config import ExperimentConfig
from src.experiments import FeatureDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description='Build and export the feature dataset.')
    parser.add_argument('--dataset-root', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/csv'))
    args = parser.parse_args()

    builder = FeatureDatasetBuilder(config=ExperimentConfig())
    dataset = builder.build(args.dataset_root)
    builder.export(dataset, args.output_root)


if __name__ == "__main__":
    main()
