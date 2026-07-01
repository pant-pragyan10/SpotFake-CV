from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configs.config import ExperimentConfig, LabConfig, OUTPUTS_DIR

from ..logging_utils import get_logger
from ..registry import available_feature_names
from ..utils.io import ensure_directory, save_dataframe, save_json, save_text
from .ablation import AblationFramework
from .correlation import FeatureCorrelationAnalyzer
from .dataset_builder import FeatureDatasetBuilder
from .integrity import DatasetIntegrityAnalyzer, ExtractorAuditAnalyzer
from .importance import FeatureImportanceAnalyzer
from .latency import LatencyBenchmark
from .report import FeatureReportWriter
from .statistics import FeatureStatisticsCalculator
from .types import ExperimentArtifacts, ExperimentLabResult, FeatureDataset
from .visualization import FeatureVisualizer


@dataclass(frozen=True)
class ExperimentRunnerConfig:
    dataset_root: Path
    output_root: Path = OUTPUTS_DIR
    lab: LabConfig = field(default_factory=LabConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


class ExperimentRunner:
    def __init__(self, config: ExperimentRunnerConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.dataset_builder = FeatureDatasetBuilder(config=self.config.experiment)
        self.statistics = FeatureStatisticsCalculator()
        self.integrity = DatasetIntegrityAnalyzer()
        self.extractor_audit = ExtractorAuditAnalyzer()
        self.correlation = FeatureCorrelationAnalyzer()
        self.importance = FeatureImportanceAnalyzer()
        self.visualizer = FeatureVisualizer(self.config.output_root / 'figures')
        self.latency = LatencyBenchmark()
        self.ablation = AblationFramework()
        self.report_writer = FeatureReportWriter()

    def run(self) -> ExperimentLabResult:
        output_root = ensure_directory(self.config.output_root)
        csv_dir = ensure_directory(output_root / 'csv')
        benchmark_dir = ensure_directory(output_root / 'benchmarks')
        reports_dir = ensure_directory(output_root / 'reports')
        figures_dir = ensure_directory(output_root / 'figures')
        logs_dir = ensure_directory(output_root / 'logs')

        self.logger.info('Available feature extractors: %s', available_feature_names())
        dataset = self.dataset_builder.build(self.config.dataset_root)
        exported_paths = self.dataset_builder.export(dataset, csv_dir)
        dataset_csv = exported_paths['csv']
        dataset_npy = exported_paths['npy']
        dataset_parquet = exported_paths['parquet']

        dataset_summary = self.integrity.summarize(dataset)
        extractor_audit_summary = self.extractor_audit.summarize(dataset)
        feature_summary = None
        correlation_summary = None
        importance_summary = None
        latency_summary = None
        ablation_summary = None

        if self.config.lab.latency_only:
            latency_summary = self.latency.run(dataset)
        elif self.config.lab.visualization_only:
            self._run_visualizations(dataset)
        elif self.config.lab.statistics_only:
            feature_summary = self.statistics.summarize(dataset)
        else:
            if self.config.lab.run_statistics:
                feature_summary = self.statistics.summarize(dataset)
            if self.config.lab.run_correlation:
                correlation_summary = self.correlation.analyze(dataset)
            if self.config.lab.run_importance:
                importance_summary = self.importance.rank(dataset)
            if self.config.lab.run_ablation:
                ablation_summary = self.ablation.run(dataset)
            if self.config.lab.run_latency:
                latency_summary = self.latency.run(dataset)
            if self.config.lab.run_visualizations:
                self._run_visualizations(dataset)

        report_files: list[Path] = []
        report_files.append(save_text(dataset_summary.summary_text, reports_dir / 'dataset_statistics.md'))
        report_files.append(save_text(extractor_audit_summary.report_text, reports_dir / 'extractor_audit.md'))
        if self.config.lab.run_reports and feature_summary is not None:
            report_path = self.report_writer.write(
                reports_dir,
                feature_summary.statistics,
                correlation_summary.pearson if correlation_summary else None,
                importance_summary.ranking if importance_summary else None,
                latency_summary.benchmark if latency_summary else None,
            )
            report_files.append(report_path)
            report_files.append(save_text(feature_summary.summary_text, reports_dir / 'statistics_report.md'))
            if correlation_summary is not None:
                report_files.append(save_text(correlation_summary.report_text, reports_dir / 'correlation_report.md'))
            if importance_summary is not None:
                report_files.append(save_text(importance_summary.report_text, reports_dir / 'importance_report.md'))
            if ablation_summary is not None:
                report_files.append(save_dataframe(ablation_summary.results, reports_dir / 'ablation_report.csv'))

        benchmark_files: list[Path] = []
        if latency_summary is not None:
            benchmark_files.append(save_dataframe(latency_summary.benchmark, benchmark_dir / 'latency_benchmark.csv'))

        feature_latency_path = save_dataframe(dataset.latency_records, benchmark_dir / 'feature_latency.csv')

        logs_manifest = save_json(
            {
                'dataset_root': str(self.config.dataset_root),
                'available_extractors': available_feature_names(),
                'feature_count': len(dataset.feature_names),
                'sample_count': len(dataset.dataframe),
                'failed_images': len(dataset.failures),
                'latency_only': self.config.lab.latency_only,
                'visualization_only': self.config.lab.visualization_only,
                'statistics_only': self.config.lab.statistics_only,
            },
            logs_dir / 'run_manifest.json',
        )

        feature_statistics_path = save_text(feature_summary.summary_text if feature_summary else dataset_summary.summary_text, reports_dir / 'feature_statistics.md')
        report_files.append(feature_statistics_path)

        artifacts = ExperimentArtifacts(
            csv_files=tuple(path for path in (dataset_csv, dataset_npy, exported_paths['labels'], exported_paths['feature_names'], exported_paths['metadata'], exported_paths['failures'], dataset_parquet) if path is not None),
            figure_files=tuple(figures_dir.glob('*.png')),
            report_files=tuple(report_files),
            benchmark_files=tuple(benchmark_files + [feature_latency_path]),
            log_files=(logs_manifest,),
        )
        return ExperimentLabResult(
            dataset=dataset,
            dataset_summary=dataset_summary,
            feature_summary=feature_summary,
            correlation_summary=correlation_summary,
            importance_summary=importance_summary,
            latency_summary=latency_summary,
            extractor_audit_summary=extractor_audit_summary,
            ablation_summary=ablation_summary,
            artifacts=artifacts,
        )

    def _run_visualizations(self, dataset: FeatureDataset) -> None:
        feature_columns = list(dataset.feature_names)
        self.visualizer.histograms(dataset.dataframe, feature_columns)
        self.visualizer.box_plots(dataset.dataframe, feature_columns)
        self.visualizer.violin_plots(dataset.dataframe, feature_columns)
        self.visualizer.kde_distributions(dataset.dataframe, feature_columns)
        self.visualizer.class_wise_distributions(dataset.dataframe, feature_columns)
        if len(feature_columns) >= 2:
            pearson = dataset.dataframe[feature_columns].corr(method='pearson')
            self.visualizer.correlation_heatmap(pearson)


def run_experiment(*args, **kwargs):
    raise NotImplementedError
