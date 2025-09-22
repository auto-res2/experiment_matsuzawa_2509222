"""Model training related utilities.
Since the original monolithic script that should have contained the training
logic is missing, the functions below merely raise informative exceptions.
They are kept so that any import from other modules succeeds gracefully.

NOTE: If the upstream project later provides a proper implementation,
all TODO markers must be replaced with the real training code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Entry-point for training.

    Parameters
    ----------
    config : Dict[str, Any]
        A nested dictionary produced by `yaml.safe_load` on the
        corresponding experiment configuration file.

    Returns
    -------
    Dict[str, Any]
        A dictionary (JSON-serialisable) with the training summary that will
        be written to `.research/iteration1/` by `main.py`.
    """
    msg = (
        "train_model() was called but the upstream monolithic script that "
        "should contain the real training logic was not supplied. "
        "Aborting to avoid silent success."
    )
    logger.error(msg)
    raise NotImplementedError(msg)
