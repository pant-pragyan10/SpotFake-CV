from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable

from .feature_extraction import FeatureExtractor


_REGISTERED_EXTRACTORS: dict[str, type[FeatureExtractor]] = {}
_DISCOVERED = False


def register_feature_extractor(extractor_cls: type[FeatureExtractor]) -> type[FeatureExtractor]:
    instance = extractor_cls()
    extractor_name = instance.name()
    existing = _REGISTERED_EXTRACTORS.get(extractor_name)
    if existing is not None and existing is not extractor_cls:
        raise ValueError(f"Duplicate feature extractor name: {extractor_name}")

    _REGISTERED_EXTRACTORS[extractor_name] = extractor_cls
    return extractor_cls


def discover_feature_extractors() -> dict[str, type[FeatureExtractor]]:
    global _DISCOVERED
    if _DISCOVERED:
        return dict(_REGISTERED_EXTRACTORS)

    package = importlib.import_module("src.features")
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("__"):
            continue
        importlib.import_module(f"{package.__name__}.{module_info.name}")

    _DISCOVERED = True
    return dict(_REGISTERED_EXTRACTORS)


def available_feature_names() -> tuple[str, ...]:
    return tuple(discover_feature_extractors().keys())


def create_feature_extractors(enabled_feature_names: Iterable[str] | None = None) -> tuple[FeatureExtractor, ...]:
    enabled = None if not enabled_feature_names else set(enabled_feature_names)
    extractor_classes = discover_feature_extractors()
    extractors: list[FeatureExtractor] = []
    for extractor_name in sorted(extractor_classes):
        if enabled is not None and extractor_name not in enabled:
            continue
        extractors.append(extractor_classes[extractor_name]())
    return tuple(extractors)