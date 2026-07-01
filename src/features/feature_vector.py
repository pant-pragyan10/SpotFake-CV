from __future__ import annotations

import logging
from time import perf_counter

from ..feature_extraction import FeatureExtractor
from ..logging_utils import get_logger
from ..registry import create_feature_extractors
from ..results import FeatureResult, FeatureVectorResult
from ..timing import TimingResult, current_memory_usage_mb


class FeatureVectorBuilder:
    def __init__(
        self,
        feature_extractors: tuple[FeatureExtractor, ...] | None = None,
        enabled_feature_names: tuple[str, ...] | None = None,
        record_memory: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self._feature_extractors = (
            feature_extractors
            if feature_extractors is not None
            else create_feature_extractors(enabled_feature_names)
        )
        self._record_memory = record_memory
        self._logger = logger or get_logger(__name__)

    @property
    def feature_extractors(self) -> tuple[FeatureExtractor, ...]:
        return self._feature_extractors

    def build(self, image: object) -> FeatureVectorResult:
        start_time = perf_counter()
        feature_values: list[float] = []
        feature_names: list[str] = []
        feature_metadata = []
        extractor_results: list[FeatureResult] = []
        extractor_errors: list[FeatureResult] = []

        for extractor in self._feature_extractors:
            self._logger.info("Running %s extractor", extractor.name())
            extractor_start = perf_counter()
            names = extractor.feature_names()
            metadata = extractor.feature_metadata()
            try:
                values = extractor.validate_output(extractor.extract(image))
                if len(values) != len(names):
                    raise ValueError(
                        f"Extractor {extractor.name()} returned {len(values)} values for {len(names)} names"
                    )

                feature_values.extend(values)
                feature_names.extend(names)
                feature_metadata.extend(metadata)

                extractor_timing = TimingResult(
                    execution_time_seconds=perf_counter() - extractor_start,
                    memory_usage_mb=current_memory_usage_mb() if self._record_memory else None,
                    item_count=len(values),
                )
                extractor_results.append(
                    FeatureResult(
                        extractor_name=extractor.name(),
                        feature_names=tuple(names),
                        feature_values=tuple(values),
                        feature_metadata=tuple(metadata),
                        timing=extractor_timing,
                    )
                )
                self._logger.info("Finished %s: %d features", extractor.name(), len(values))
            except Exception as exc:
                extractor_timing = TimingResult(
                    execution_time_seconds=perf_counter() - extractor_start,
                    memory_usage_mb=current_memory_usage_mb() if self._record_memory else None,
                    item_count=0,
                )
                error_message = f"{type(exc).__name__}: {exc}"
                extractor_errors.append(
                    FeatureResult(
                        extractor_name=extractor.name(),
                        feature_names=tuple(names),
                        feature_values=(),
                        feature_metadata=tuple(metadata),
                        timing=extractor_timing,
                        error=error_message,
                    )
                )
                self._logger.exception("Extractor %s failed", extractor.name())

        timing = TimingResult(
            execution_time_seconds=perf_counter() - start_time,
            memory_usage_mb=current_memory_usage_mb() if self._record_memory else None,
            item_count=len(feature_values),
        )
        self._logger.info("Feature vector built with %d total features", len(feature_values))
        return FeatureVectorResult(
            feature_vector=tuple(feature_values),
            feature_names=tuple(feature_names),
            feature_metadata=tuple(feature_metadata),
            timing=timing,
            extractor_results=tuple(extractor_results),
            extractor_errors=tuple(extractor_errors),
        )
