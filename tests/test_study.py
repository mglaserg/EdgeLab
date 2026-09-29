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


    def test_evidence_export_is_conductor_compatible_and_fail_closed_on_window(self):
        from edgelab.evidence import EvidenceRecord
        from edgelab.evaluate import Evaluation
        from edgelab.priors import CrisisState, EconomicPrior
        from edgelab.registry import Hypothesis

        hyp = Hypothesis(
            id="hyp123",
            family="family",
            name="candidate",
            prior=EconomicPrior(
                tier=PriorTier.CONSTRAINT,
                counterparty=(
                    "Mandated funds rebalance mechanically at a known calendar boundary."
                ),
                persistence=(
                    "The mandate creates recurring price-insensitive flow that cannot simply opt out."
                ),
                crisis_state=CrisisState.IDIOSYNCRATIC,
                falsifier=(
                    "The return effect disappears on the preregistered event calendar out of sample."
                ),
            ),
            estimand="mean daily net PnL",
            universe="synthetic assets",
            feature_spec="candidate@v1",
            is_end="2024-12-31",
            oos_start="2025-01-01",
            registered_at="2025-01-01T00:00:00+00:00",
        )
        ev = Evaluation(
            hypothesis_id=hyp.id,
            family=hyp.family,
            variant="baseline",
            window="oos",
            tier=hyp.prior.tier,
            n_obs=250,
            mean_pnl=0.001,
            sharpe_per_obs=0.1,
            sharpe_annualised=1.58,
            skew=0.0,
            excess_kurtosis=0.0,
            mean_ci=(0.0002, 0.0018),
            sharpe_ci=(0.3, 2.2),
            block_length=6.0,
            raw_trials=2,
            effective_trials=3.0,
            sharpe_hurdle=0.02,
            dsr=0.99,
            trial_id="trial123",
        )
        record = EvidenceRecord.from_evaluation(hyp, ev)
        self.assertEqual(record.decision, "pass")
        self.assertTrue(record.eligible_for_promotion)
        self.assertEqual(record.evaluation["trial_id"], "trial123")
        self.assertEqual(
            record.conductor_reference("artifacts/evidence.json"),
            {
                "producer": "edgelab",
                "artifact_type": "validation",
                "location": "artifacts/evidence.json",
                "version": "0.3.0",
            },
        )

        diagnostic = Evaluation(**{**ev.__dict__, "window": "training"})
        diagnostic_record = EvidenceRecord.from_evaluation(hyp, diagnostic)
        self.assertEqual(diagnostic_record.decision, "diagnostic")
        self.assertFalse(diagnostic_record.eligible_for_promotion)

    def test_study_evidence_redeflates_against_final_family_trial_count(self):
        rng = np.random.default_rng(7)
        pnl_a = rng.normal(0.001, 0.01, size=180)
        pnl_b = rng.normal(0.001, 0.01, size=180)

        with TemporaryDirectory() as tmp:
            with edgelab.study(
                name="evidence test",
                idea="A candidate edge earns positive net returns out of sample.",
                universe="synthetic daily asset",
                target="mean daily net PnL",
                is_end="2024-12-31",
                oos_start="2025-01-01",
                db=Path(tmp) / "edgelab.db",
            ) as s:
                first = s.test(pnl_a, variant="a", n_boot=50, seed=1)
                second = s.test(pnl_b, variant="b", n_boot=50, seed=2)
                self.assertTrue(first.trial_id)
                self.assertTrue(second.trial_id)
                self.assertEqual(first.raw_trials, 1)
                self.assertEqual(second.raw_trials, 2)

                out = Path(tmp) / "artifacts" / "validation.json"
                record = s.write_evidence(first, out)
                self.assertTrue(out.exists())
                self.assertEqual(record.evaluation["raw_trials"], 2)
                self.assertIn(
                    "Re-deflated against the full family history",
                    " ".join(record.evaluation["warnings"]),
                )

                import json
                payload = json.loads(out.read_text())
                self.assertEqual(payload["evidence_id"], record.evidence_id)
                self.assertEqual(payload["schema_version"], "edgelab.validation.v1")


if __name__ == "__main__":
    unittest.main()
