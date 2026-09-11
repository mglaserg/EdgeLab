"""edgelab -- pre-registered edge validation.

The spine, not the search.  The easiest path is now::

    s = edgelab.study(...)
    d = s.diagnose(signal, returns)
    ev = s.test(net_pnl)

The lower-level mechanism, diagnostics, registry and evaluation APIs remain
available when you need full control.
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
from .study import Study, StudyDiagnostics, study, unlabeled_prior

__version__ = "0.2.0"

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
    "Study", "StudyDiagnostics", "study", "unlabeled_prior",
]
