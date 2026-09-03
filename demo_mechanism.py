"""Two strategies claim to be risk premia. Only one is.

The point: both authors filled in the counterparty field convincingly. Only
the data decides which story survives.
"""

import numpy as np

from edgelab.mechanism import (
    AgentType, BarrierType, Compulsion, Counterparty, Mechanism, Persistence,
    test_calendar_concentration, test_crisis_loss, test_no_crowding_decay,
    test_proxy_conditioning,
)
from edgelab.priors import CrisisState

rng = np.random.default_rng(11)
N = 1000
stress = rng.random(N) < 0.08          # 8% of days are vol spikes

# =========================================================================
# A. genuine short-vol premium: earns in calm, bleeds hard in stress
# =========================================================================
flow_a = rng.normal(0, 1, N)           # hedging demand, standardised
pnl_a = np.where(stress,
                 rng.normal(-0.010, 0.006, N),
                 rng.normal(0.0009, 0.004, N) + 0.0010 * flow_a)
# richer insurance demand -> more premium collected, which is the mechanism

mech_a = Mechanism(
    counterparty=Counterparty(
        agent=AgentType.INSURANCE_BUYER,
        compulsion=Compulsion.HEDGING_NEED,
        direction="buys index puts to cover mandated downside protection",
        observable_proxy="SPX put OI / 20d ADV, front two expiries",
    ),
    persistence=Persistence(
        barriers=(BarrierType.RISK_TOLERANCE, BarrierType.CAPITAL_CHARGE),
        capacity_usd=250_000_000,
        rationale="supplying tail insurance needs balance sheet and tolerance "
                  "for path-dependent losses; most allocators cannot hold it",
    ),
    crisis_state=CrisisState.VOL_SPIKE,
)
mech_a.record(
    test_crisis_loss(pnl_a, stress),
    test_proxy_conditioning(pnl_a, flow_a),
    test_no_crowding_decay(pnl_a),
)
print(mech_a.report(), "\n")

# =========================================================================
# B. same story told, but the PnL does not lose in the bad state
# =========================================================================
pnl_b = rng.normal(0.0006, 0.005, N)   # edge unrelated to the claimed risk
flow_b = rng.normal(0, 1, N)           # proxy is noise

mech_b = Mechanism(
    counterparty=Counterparty(
        agent=AgentType.INSURANCE_BUYER,
        compulsion=Compulsion.HEDGING_NEED,
        direction="buys index puts for downside protection",
        observable_proxy="SPX put/call open interest ratio",
    ),
    persistence=Persistence(
        barriers=(BarrierType.RISK_TOLERANCE,),
        capacity_usd=100_000_000,
        rationale="same story as A -- told just as fluently",
    ),
    crisis_state=CrisisState.VOL_SPIKE,
)
mech_b.record(
    test_crisis_loss(pnl_b, stress),
    test_proxy_conditioning(pnl_b, flow_b),
    test_no_crowding_decay(pnl_b),
)
print(mech_b.report(), "\n")

# =========================================================================
# C. a constraint claim whose edge ignores its own calendar
# =========================================================================
day = np.arange(N) % 21
month_end = day >= 19
pnl_c = rng.normal(0.0005, 0.004, N)   # uniform -- NOT concentrated

mech_c = Mechanism(
    counterparty=Counterparty(
        agent=AgentType.MANDATED_FUND,
        compulsion=Compulsion.MANDATE,
        direction="rebalances to target weights at month end",
        observable_proxy="month-end index rebalance notional estimate",
        calendar="last two trading days of each month",
    ),
    persistence=Persistence(
        barriers=(BarrierType.CAPACITY,),
        capacity_usd=40_000_000,
    ),
    crisis_state=CrisisState.IDIOSYNCRATIC,
)
mech_c.record(test_calendar_concentration(pnl_c, month_end))
print(mech_c.report())
