"""Demonstration: the harness must kill a winner mined from pure noise.

Scenario A -- forty zero-edge variants are searched, the best is reported.
Scenario B -- a genuine, modest edge with a real risk-premium story.

If A survives and B does not, the harness is worthless. Run it and check.
"""

import numpy as np

from edgelab import CrisisState, EconomicPrior, PriorTier, Registry, evaluate
from edgelab.registry import UnregisteredHypothesis

rng = np.random.default_rng(7)
reg = Registry("/tmp/demo_edgelab.db")

# -- guard rail: no result without pre-registration ------------------------
try:
    evaluate(reg, "does_not_exist", rng.normal(size=500),
             variant="v0", window="oos")
except UnregisteredHypothesis as exc:
    print("REFUSED as designed:\n ", str(exc)[:90], "...\n")

# =========================================================================
# A. mined noise, dressed up with a story
# =========================================================================
mined = reg.register(
    family="reversal_mined",
    name="5d reversal, parameter swept",
    prior=EconomicPrior(
        tier=PriorTier.UNLABELED,
        counterparty="unclear -- no identified forced seller in this window",
        persistence="no mechanism proposed; found by sweeping lookback params",
        crisis_state=CrisisState.UNKNOWN,
        falsifier="should vanish out of sample if it is a sweep artifact",
    ),
    estimand="mean daily net PnL of the reversal portfolio",
    universe="US large cap, 2015-2024",
    feature_spec="reversal@v1",
    is_end="2021-12-31",
    oos_start="2022-01-01",
)

best, best_sr = None, -np.inf
for k in range(40):                       # every sweep gets logged
    noise = rng.normal(0.0, 0.01, 750)
    ev = evaluate(reg, mined.id, noise, variant=f"lookback={k+3}",
                  window="oos", n_boot=800, seed=k)
    if ev.sharpe_annualised > best_sr:
        best, best_sr = ev, ev.sharpe_annualised

print("A. BEST OF 40 ZERO-EDGE SWEEPS")
print(best.summary(), "\n")

# =========================================================================
# B. a real, modest edge with a real story
# =========================================================================
vrp = reg.register(
    family="spy_vrp",
    name="Short SPY variance vs realised",
    prior=EconomicPrior(
        tier=PriorTier.RISK_PREMIUM,
        counterparty="hedgers and mandated overwriters buying index protection",
        persistence="insurance demand is structural and price-insensitive; "
                    "supplying it requires tolerating gap risk",
        crisis_state=CrisisState.VOL_SPIKE,
        falsifier="must lose materially in any month with a >5 vol point spike",
    ),
    estimand="mean daily net PnL of the short-variance sleeve",
    universe="SPY options, 2016-2024",
    feature_spec="vrp@v3",
    is_end="2021-12-31",
    oos_start="2022-01-01",
)

# modest edge, negatively skewed -- what short vol actually looks like
real = 0.0004 + 0.008 * rng.standard_t(4, 750) / np.sqrt(2.0)
real -= 0.004 * (rng.random(750) < 0.02)          # occasional gap losses
ev_real = evaluate(reg, vrp.id, real, variant="delta16_weekly",
                   window="oos", n_boot=3000, seed=99)

print("B. GENUINE EDGE, ONE PRE-REGISTERED VARIANT")
print(ev_real.summary(), "\n")

print("Crisis-state concentration across the book:", reg.crisis_exposure())
reg.close()

# =========================================================================
# C. the same mined winner, scored against the FULL sweep
# =========================================================================
from edgelab import redeflate
reg2 = Registry("/tmp/demo_edgelab.db")
print("\nC. RE-DEFLATED AGAINST THE COMPLETE SEARCH")
print(redeflate(reg2, best).summary())
reg2.close()
