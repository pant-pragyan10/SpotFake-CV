from __future__ import annotations

import logging


def configure_logging(enabled: bool = True, level: str = "INFO") -> None:
    if not enabled:
        logging.disable(logging.CRITICAL)
        return

    logging.disable(logging.NOTSET)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)