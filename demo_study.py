"""The simple EdgeLab workflow: idea -> diagnose -> test.

No mechanism is required up front.  If you do not know why the edge exists,
EdgeLab registers it honestly as UNLABELED and applies the stronger haircut.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import edgelab

rng = np.random.default_rng(42)
n = 900

# A weak toy signal: today's signal has a little information about tomorrow.
signal = rng.normal(size=n)
returns = rng.normal(0.0002, 0.01, size=n)
returns[1:] += 0.00055 * signal[:-1]

with TemporaryDirectory() as tmp:
    with edgelab.study(
        name="Toy one-day signal",
        idea="Higher signal values predict higher next-day returns.",
        why="I do not know yet; this is a statistical lead, not a mechanism claim.",
        universe="Synthetic daily asset returns",
        target="mean daily net PnL from trading the signal",
        feature_spec="toy_signal@v1",
        is_end="2024-12-31",
        oos_start="2025-01-01",
        db=Path(tmp) / "edgelab.db",
    ) as s:
        print(s.summary(), "\n")

        d = s.diagnose(signal, returns, cost_per_turnover=0.00005)
        print(d.summary(), "\n")

        ev = s.test(d.portfolio.net_returns, variant="baseline", n_boot=1200, seed=7)
        print(ev.summary())
