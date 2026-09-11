import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import edgelab
from edgelab.priors import PriorTier


class StudyTests(unittest.TestCase):
    def test_simple_study_defaults_to_unlabeled_and_runs(self):
        rng = np.random.default_rng(1)
        n = 350
        signal = rng.normal(size=n)
        returns = rng.normal(0, 0.01, size=n)
        returns[1:] += 0.0004 * signal[:-1]

        with TemporaryDirectory() as tmp:
            s = edgelab.study(
                name="simple test",
                idea="Higher signal predicts higher next-day returns",
                universe="synthetic daily asset",
                target="mean daily net PnL",
                is_end="2024-12-31",
                oos_start="2025-01-01",
                db=Path(tmp) / "nested" / "edgelab.db",
            )
            try:
                self.assertEqual(s.hypothesis.prior.tier, PriorTier.UNLABELED)
                d = s.diagnose(signal, returns, n_boot=100)
                self.assertTrue(np.isfinite(d.portfolio.realised_vol))
                ev = s.test(d.portfolio.net_returns, n_boot=100, seed=2)
                self.assertEqual(ev.raw_trials, 1)
                self.assertEqual(s.registry.trial_count(s.family), 1)
            finally:
                s.close()

    def test_plain_why_does_not_upgrade_prior(self):
        with TemporaryDirectory() as tmp:
            s = edgelab.study(
                name="story test",
                idea="This is a testable trading idea",
                why="Maybe institutions are forced to rebalance every month end",
                universe="synthetic market",
                target="next-day return",
                is_end="2024-12-31",
                oos_start="2025-01-01",
                db=Path(tmp) / "x.db",
            )
            try:
                self.assertEqual(s.hypothesis.prior.tier, PriorTier.UNLABELED)
                self.assertIn("unverified", s.hypothesis.prior.persistence.lower())
            finally:
                s.close()


    def test_decay_profile_uses_immediate_forward_window(self):
        rng = np.random.default_rng(123)
        n = 500
        signal = rng.normal(size=n)
        returns = rng.normal(scale=0.01, size=n)
        returns[1:] = signal[:-1] + rng.normal(scale=0.05, size=n - 1)
        decay = edgelab.decay_profile(signal, returns, horizons=(1, 2))
        self.assertGreater(decay[1], 0.95)
        self.assertGreater(decay[2], 0.50)

    def test_registry_creates_parent_directory(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "a" / "b" / "registry.db"
            reg = edgelab.Registry(db)
            try:
                self.assertTrue(db.parent.exists())
            finally:
                reg.close()


if __name__ == "__main__":
    unittest.main()
