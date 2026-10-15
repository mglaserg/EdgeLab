from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import edgelab
from edgelab.evaluate import deflated_sharpe


def test_large_search_deflates_apparent_discovery():
    # A modest selected Sharpe can look compelling after two tries, but not
    # after the thousand related configurations that actually produced it.
    small_search, _ = deflated_sharpe(0.15, 250, 0.0, 3.0, 2, 1 / 249)
    large_search, _ = deflated_sharpe(0.15, 250, 0.0, 3.0, 1000, 1 / 249)
    assert small_search > 0.95
    assert large_search < 0.95


def test_dsr_uses_pearson_not_excess_kurtosis():
    rng = np.random.default_rng(72)
    pnl = rng.normal(0.001, 0.01, 500)
    with TemporaryDirectory() as tmp:
        with edgelab.study(
            name="kurtosis check",
            idea="A synthetic candidate has positive average net PnL.",
            universe="synthetic daily asset",
            target="mean daily net PnL",
            is_end="2024-12-31",
            oos_start="2025-01-01",
            db=Path(tmp) / "edge.db",
        ) as study:
            ev = study.test(pnl, window="training", n_boot=100, seed=1)
            expected, _ = deflated_sharpe(
                ev.sharpe_per_obs,
                ev.n_obs,
                ev.skew,
                ev.excess_kurtosis + 3.0,
                ev.effective_trials,
                1.0 / (ev.n_obs - 1),
            )
            assert ev.dsr == pytest.approx(expected)


def test_holdout_claim_requires_and_checks_observation_times():
    rng = np.random.default_rng(9)
    pnl = rng.normal(0.001, 0.01, 60)
    with TemporaryDirectory() as tmp:
        with edgelab.study(
            name="sealed dates",
            idea="A sealed dated sample tests a registered candidate.",
            universe="synthetic daily asset",
            target="mean daily net PnL",
            is_end="2024-12-31",
            oos_start="2025-01-01",
            db=Path(tmp) / "edge.db",
        ) as study:
            unverified = study.test(pnl, window="oos", n_boot=50, seed=1)
            assert not unverified.holdout_verified
            assert not study.evidence(unverified).eligible_for_promotion

            dates = np.arange("2025-01-01", "2025-03-02", dtype="datetime64[D]")
            verified = study.test(
                pnl, window="oos", observation_times=dates, n_boot=50, seed=2
            )
            assert verified.holdout_verified

            bad_dates = dates.copy()
            bad_dates[0] = np.datetime64("2024-12-31")
            with pytest.raises(ValueError, match="before preregistered"):
                study.test(
                    pnl, window="oos", observation_times=bad_dates,
                    n_boot=20, seed=3,
                )


def test_registry_rejects_non_iso_dates():
    with TemporaryDirectory() as tmp:
        registry = edgelab.Registry(Path(tmp) / "edge.db")
        try:
            with pytest.raises(ValueError, match="ISO dates"):
                edgelab.Study.create(
                    registry=registry,
                    name="bad dates",
                    idea="This candidate has malformed registration dates.",
                    universe="synthetic market",
                    target="mean PnL",
                    is_end="2024-2-1",
                    oos_start="2024-10-01",
                )
        finally:
            registry.close()
