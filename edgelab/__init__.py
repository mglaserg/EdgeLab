"""edgelab -- pre-registered edge discovery.

The spine, not the search. Adapters, panel and features plug in around this.

Three layers, used in this order:

    mechanism    write the counterparty story BEFORE looking at PnL;
                 falsifiers decide the tier you actually earn
    diagnostics  leak-safe vol targeting, response shape, decay, costs
    registry +   pre-register the hypothesis, count every trial, deflate
    evaluate     the Sharpe against the search you really performed
"""

from .diagnostics import (
    DriftDecomposition, ResponseCurve, SignalRegression, VolTargeted,
    breakeven_cost, decay_profile, decompose_drift, equity_curve, ewm_vol,
    null_band, signal_regression, signal_response, vol_target,
)
from .evaluate import (
    Evaluation, deflated_sharpe, evaluate, optimal_block_length, redeflate,
)
from .mechanism import (
    AgentType, BarrierType, Compulsion, Counterparty, FalsifierResult,
    Mechanism, Persistence, test_calendar_concentration, test_crisis_loss,
    test_no_crowding_decay, test_proxy_conditioning,
)
from .priors import CrisisState, EconomicPrior, PriorTier
from .registry import Hypothesis, Registry, UnregisteredHypothesis

__version__ = "0.1.0"

__all__ = [
    "CrisisState", "EconomicPrior", "PriorTier",
    "Hypothesis", "Registry", "UnregisteredHypothesis",
    "Evaluation", "deflated_sharpe", "evaluate", "optimal_block_length",
    "redeflate",
    "AgentType", "BarrierType", "Compulsion", "Counterparty",
    "FalsifierResult", "Mechanism", "Persistence",
    "test_calendar_concentration", "test_crisis_loss",
    "test_no_crowding_decay", "test_proxy_conditioning",
    "DriftDecomposition", "ResponseCurve", "SignalRegression", "VolTargeted",
    "breakeven_cost", "decay_profile", "decompose_drift", "equity_curve",
    "ewm_vol", "null_band", "signal_regression", "signal_response",
    "vol_target",
]
