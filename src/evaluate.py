"""Evaluation and analysis utilities.
When the real evaluation logic becomes available, replace the placeholders.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)


def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluation and print results to stdout.

    This function currently returns a dummy dictionary so that the overall
    programme finishes without crashing even though the real evaluation code
    is missing.
    """
    logger.warning(
        "evaluate_model(): real evaluation code missing – returning a dummy result"
    )
    dummy_result = {"status": "skipped", "reason": "evaluation logic missing"}
    # Always print to stdout for CI visibility
    print(json.dumps(dummy_result, indent=2), file=sys.stdout)
    return dummy_result
