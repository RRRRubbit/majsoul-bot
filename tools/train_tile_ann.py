"""兼容入口：转发到 majsoul_bot.tools.train_tile_ann。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from majsoul_bot.tools.train_tile_ann import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
