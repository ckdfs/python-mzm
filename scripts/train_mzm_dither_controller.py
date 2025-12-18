"""CLI entry for training + quick rollout sanity check.

The reusable implementation lives in mzm.dither_controller.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mzm.dither_controller import main


if __name__ == "__main__":
    raise SystemExit(main())
