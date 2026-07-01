from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from configs.config import PipelineConfig

from .features.feature_vector import FeatureVectorBuilder
from .logging_utils import configure_logging, get_logger
from .preprocessing.loader import load_image
from .preprocessing.normalize import normalize_image
from .preprocessing.resize import resize_image
from .results import ImageInfo, PipelineResult, PreprocessingResult
from .timing import TimingResult, current_memory_usage_mb


class ExperimentPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        feature_vector_builder: FeatureVectorBuilder | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        configure_logging(self.config.logging.enabled, self.config.logging.level)
        self.logger = logger or get_logger(__name__)
        self.feature_vector_builder = feature_vector_builder or FeatureVectorBuilder(
            enabled_feature_names=self.config.features.enabled_feature_extractors,
            record_memory=self.config.timing.record_memory,
            logger=self.logger,
        )

    def preprocess(self, image):
        start_time = perf_counter()
        steps: list[str] = []
        processed_image = image

        if self.config.preprocessing.enabled:
            if self.config.preprocessing.resize_enabled:
                processed_image = resize_image(
                    processed_image,
                    (self.config.preprocessing.resize_width, self.config.preprocessing.resize_height),
                )
                steps.append("resize")
            processed_image = normalize_image(processed_image)
            steps.append("normalize")

        timing = TimingResult(
            execution_time_seconds=perf_counter() - start_time,
            memory_usage_mb=current_memory_usage_mb() if self.config.timing.record_memory else None,
            item_count=len(steps),
        )
        return processed_image, PreprocessingResult(
            original_size=image.size,
            processed_size=processed_image.size,
            applied_steps=tuple(steps),
            timing=timing,
        )

    def run(self, image_path: str | Path) -> PipelineResult:
        path = Path(image_path)
        self.logger.info("Loaded image %s", path)
        image = load_image(path)
        image_info = ImageInfo(path=path, width=image.width, height=image.height, mode=image.mode)

        preprocessed_image, preprocessing_result = self.preprocess(image)
        self.logger.info("Running feature extraction")
        feature_vector_result = self.feature_vector_builder.build(preprocessed_image)

        total_timing = TimingResult(
            execution_time_seconds=
            preprocessing_result.timing.execution_time_seconds
            + feature_vector_result.timing.execution_time_seconds,
            memory_usage_mb=current_memory_usage_mb() if self.config.timing.record_memory else None,
            item_count=len(feature_vector_result.feature_vector),
        )
        self.logger.info("Pipeline completed with %d features", len(feature_vector_result.feature_vector))
        return PipelineResult(
            image_info=image_info,
            preprocessing=preprocessing_result,
            feature_vector=feature_vector_result,
            probability=0.5,
            timing=total_timing,
        )