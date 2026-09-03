"""Mechanism: the counterparty story, made structural and testable.

Free-text "who is on the other side" gets written to satisfy the form. This
module replaces it with three commitments that are hard to fake:

1. A typed agent and the specific compulsion that forces them to trade.
2. An OBSERVABLE PROXY -- a named data series that measures the flow. If you
   cannot name a series, you do not have a mechanism, you have a vibe.
3. Barriers explaining why arbitrage capital has not closed the gap, with a
   capacity number attached.

The tier is then DERIVED from what you filled in and, crucially, capped by
whether your own falsifier survived contact with data. Asserting
RISK_PREMIUM does not make it one; losing money in the named crisis state does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .priors import CrisisState, PriorTier


class AgentType(str, Enum):
    """Who is on the other side. 'The market' is not an answer."""

    DEALER = "dealer"                      # hedging inventory, gamma/vanna flow
    INDEX_TRACKER = "index_tracker"        # must hold the benchmark
    MANDATED_FUND = "mandated_fund"        # overwriters, target-vol, risk parity
    INSURANCE_BUYER = "insurance_buyer"    # pays to transfer tail risk
    LEVERAGED_HOLDER = "leveraged_holder"  # forced out on margin
    RETAIL = "retail"                      # weakest claim; use sparingly
    CORPORATE = "corporate"                # buybacks, hedging programmes
    ARBITRAGEUR = "arbitrageur"            # if this is your counterparty, reconsider


class Compulsion(str, Enum):
    """WHY they trade regardless of price. This is the load-bearing field."""

    MANDATE = "mandate"                # prospectus/IPS requires it
    REGULATION = "regulation"          # capital or reporting rules
    MARGIN = "margin"                  # forced liquidation
    BENCHMARK_TRACKING = "benchmark_tracking"
    RISK_LIMIT = "risk_limit"          # VaR/vega/delta limits bind
    HEDGING_NEED = "hedging_need"      # real exposure to offset
    TAX = "tax"
    PREFERENCE = "preference"          # NOT a compulsion -- flags behavioural


class BarrierType(str, Enum):
    """Why hasn't this been arbitraged away?"""

    CAPITAL_CHARGE = "capital_charge"
    BALANCE_SHEET = "balance_sheet"
    BORROW_CONSTRAINT = "borrow_constraint"
    SEGMENTATION = "segmentation"        # different investor pools
    CAPACITY = "capacity"                # too small to interest big capital
    OPERATIONAL = "operational"          # infrastructure/data cost
    RISK_TOLERANCE = "risk_tolerance"    # arbs CAN but won't hold this risk
    NONE_IDENTIFIED = "none_identified"  # honest, and heavily penalised


@dataclass(frozen=True)
class Counterparty:
    agent: AgentType
    compulsion: Compulsion
    direction: str            # "buys downside protection into expiry"
    observable_proxy: str     # NAMED SERIES, e.g. "CBOE put/call OI ratio"
    calendar: str | None = None   # for CONSTRAINT claims: when does flow occur

    def __post_init__(self) -> None:
        if not self.observable_proxy or len(self.observable_proxy.strip()) < 8:
            raise ValueError(
                "observable_proxy must name a data series you can actually "
                "pull. If the flow is not measurable, the mechanism is not "
                "testable and this is an unlabeled regularity."
            )

    @property
    def is_forced(self) -> bool:
        return self.compulsion is not Compulsion.PREFERENCE


@dataclass(frozen=True)
class Persistence:
    barriers: tuple[BarrierType, ...]
    capacity_usd: float          # how much can this absorb before it decays?
    expected_half_life_years: float | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.barriers:
            raise ValueError("Name at least one barrier, or NONE_IDENTIFIED.")
        if self.capacity_usd <= 0:
            raise ValueError(
                "Estimate capacity in USD. An edge with no capacity estimate "
                "cannot be sized, and unsized edges become oversized ones."
            )

    @property
    def has_real_barrier(self) -> bool:
        return BarrierType.NONE_IDENTIFIED not in self.barriers


# ---------------------------------------------------------------------------
# executable falsifiers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FalsifierResult:
    name: str
    passed: bool
    detail: str
    statistic: float = float("nan")


def _boot_mean_ci(x: np.ndarray, rng, n_boot=2000, alpha=0.05):
    if len(x) < 5:
        return float("nan"), float("nan")
    draws = np.array([x[rng.integers(0, len(x), len(x))].mean()
                      for _ in range(n_boot)])
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def test_crisis_loss(
    pnl: np.ndarray, stress_mask: np.ndarray, *, seed: int = 0
) -> FalsifierResult:
    """RISK_PREMIUM claim: you MUST lose money in the named bad state.

    A premium with no bad state is either mislabeled or the sample never
    contained the risk. Both mean you should not be paid for bearing it.
    """
    rng = np.random.default_rng(seed)
    stressed = np.asarray(pnl)[np.asarray(stress_mask, bool)]
    calm = np.asarray(pnl)[~np.asarray(stress_mask, bool)]
    if len(stressed) < 5:
        return FalsifierResult(
            "crisis_loss", False,
            f"Only {len(stressed)} stress observations. The bad state is "
            f"absent from your sample -- you have not tested the premium.",
        )
    lo, hi = _boot_mean_ci(stressed, rng)
    mean_s, mean_c = float(stressed.mean()), float(calm.mean())
    passed = hi < 0 and mean_s < mean_c
    return FalsifierResult(
        "crisis_loss", passed,
        f"stress mean {mean_s:+.5f} (CI {lo:+.5f},{hi:+.5f}) vs calm "
        f"{mean_c:+.5f}. " + (
            "Loses in the bad state, consistent with a premium."
            if passed else
            "Does NOT reliably lose in the named crisis state -- the "
            "risk-premium label is unsupported."
        ),
        statistic=mean_s,
    )


def test_calendar_concentration(
    pnl: np.ndarray, on_calendar: np.ndarray, *, seed: int = 0
) -> FalsifierResult:
    """CONSTRAINT claim: the edge must concentrate when the flow occurs.

    If a month-end effect pays equally off month-end, it is not month-end flow.
    """
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl)
    mask = np.asarray(on_calendar, bool)
    on, off = pnl[mask], pnl[~mask]
    if len(on) < 5 or len(off) < 5:
        return FalsifierResult(
            "calendar_concentration", False,
            f"Insufficient split ({len(on)} on / {len(off)} off).",
        )
    diffs = np.array([
        on[rng.integers(0, len(on), len(on))].mean()
        - off[rng.integers(0, len(off), len(off))].mean()
        for _ in range(2000)
    ])
    lo = float(np.percentile(diffs, 5))
    passed = lo > 0
    return FalsifierResult(
        "calendar_concentration", passed,
        f"on-calendar {on.mean():+.5f} vs off {off.mean():+.5f}, "
        f"one-sided 95% lower bound on the difference {lo:+.5f}. " + (
            "Edge concentrates on the claimed calendar."
            if passed else
            "Edge does NOT concentrate on-calendar -- the constraint story "
            "does not explain the return."
        ),
        statistic=lo,
    )


def test_proxy_conditioning(
    pnl: np.ndarray, proxy: np.ndarray, *, seed: int = 0
) -> FalsifierResult:
    """The observable flow proxy must actually condition returns.

    If your claimed flow measure carries no information about when the edge
    pays, you have not identified the mechanism -- you have decorated a result.
    """
    rng = np.random.default_rng(seed)
    pnl, proxy = np.asarray(pnl), np.asarray(proxy)
    ok = np.isfinite(pnl) & np.isfinite(proxy)
    pnl, proxy = pnl[ok], proxy[ok]
    if len(pnl) < 40:
        return FalsifierResult("proxy_conditioning", False,
                               f"Only {len(pnl)} paired observations.")
    hi_mask = proxy >= np.median(proxy)
    hi, lo_ = pnl[hi_mask], pnl[~hi_mask]
    diffs = np.array([
        hi[rng.integers(0, len(hi), len(hi))].mean()
        - lo_[rng.integers(0, len(lo_), len(lo_))].mean()
        for _ in range(2000)
    ])
    lb = float(np.percentile(diffs, 5))
    passed = lb > 0
    return FalsifierResult(
        "proxy_conditioning", passed,
        f"high-flow {hi.mean():+.5f} vs low-flow {lo_.mean():+.5f}, "
        f"lower bound {lb:+.5f}. " + (
            "Flow proxy conditions the return as the mechanism predicts."
            if passed else
            "Flow proxy carries no information -- mechanism unsupported."
        ),
        statistic=lb,
    )


def test_no_crowding_decay(pnl: np.ndarray, n_blocks: int = 4) -> FalsifierResult:
    """Capacity check: is the edge monotonically dying across sub-periods?

    Not a failure on its own -- decay is what a real, discovered edge does.
    But it must be known before sizing.
    """
    pnl = np.asarray(pnl)
    blocks = np.array_split(pnl, n_blocks)
    means = [float(b.mean()) for b in blocks]
    monotone_down = all(a > b for a, b in zip(means, means[1:]))
    return FalsifierResult(
        "crowding_decay", not monotone_down,
        "sub-period means " + ", ".join(f"{m:+.5f}" for m in means) + ". " + (
            "Monotone decay across every sub-period -- assume crowding and "
            "size on the most recent block, not the full sample."
            if monotone_down else "No monotone decay pattern."
        ),
        statistic=means[-1] - means[0],
    )


# ---------------------------------------------------------------------------
# the mechanism, with a derived tier
# ---------------------------------------------------------------------------

@dataclass
class Mechanism:
    counterparty: Counterparty
    persistence: Persistence
    crisis_state: CrisisState
    results: list[FalsifierResult] = field(default_factory=list)

    def record(self, *results: FalsifierResult) -> "Mechanism":
        self.results.extend(results)
        return self

    @property
    def failed(self) -> list[FalsifierResult]:
        return [r for r in self.results if not r.passed]

    def asserted_tier(self) -> PriorTier:
        """What the STRUCTURE of your story implies, before testing it."""
        cp, ps = self.counterparty, self.persistence

        if not ps.has_real_barrier:
            return PriorTier.UNLABELED
        if cp.compulsion is Compulsion.PREFERENCE:
            return PriorTier.BEHAVIORAL

        if cp.compulsion in (Compulsion.HEDGING_NEED, Compulsion.RISK_LIMIT) \
                and cp.agent in (AgentType.INSURANCE_BUYER, AgentType.DEALER,
                                 AgentType.MANDATED_FUND) \
                and self.crisis_state is not CrisisState.UNKNOWN:
            return PriorTier.RISK_PREMIUM

        if cp.compulsion in (Compulsion.MANDATE, Compulsion.BENCHMARK_TRACKING,
                             Compulsion.MARGIN, Compulsion.TAX,
                             Compulsion.REGULATION) and cp.calendar:
            return PriorTier.CONSTRAINT

        if BarrierType.CAPITAL_CHARGE in ps.barriers \
                or BarrierType.BALANCE_SHEET in ps.barriers \
                or BarrierType.SEGMENTATION in ps.barriers \
                or BarrierType.BORROW_CONSTRAINT in ps.barriers:
            return PriorTier.FRICTION

        return PriorTier.UNLABELED

    def tier(self) -> tuple[PriorTier, list[str]]:
        """The tier you actually get: asserted, then capped by evidence.

        This is the whole point. A story you cannot defend against your own
        falsifier does not earn the prior mass that story would carry.
        """
        tier = self.asserted_tier()
        notes: list[str] = [f"asserted tier from structure: {tier.value}"]

        for r in self.failed:
            if r.name == "crisis_loss" and tier is PriorTier.RISK_PREMIUM:
                tier = PriorTier.UNLABELED
                notes.append(
                    "DOWNGRADED to unlabeled: claimed a risk premium but does "
                    "not lose in the named crisis state."
                )
            elif r.name == "calendar_concentration" and tier is PriorTier.CONSTRAINT:
                tier = PriorTier.UNLABELED
                notes.append(
                    "DOWNGRADED to unlabeled: edge does not concentrate on the "
                    "claimed flow calendar."
                )
            elif r.name == "proxy_conditioning":
                if tier in (PriorTier.RISK_PREMIUM, PriorTier.CONSTRAINT):
                    tier = PriorTier.FRICTION
                    notes.append(
                        "DOWNGRADED to friction: the named flow proxy does not "
                        "condition returns, so the mechanism is unverified."
                    )
            elif r.name == "crowding_decay":
                notes.append(
                    "Crowding decay detected -- tier unchanged, but size on "
                    "recent performance only."
                )
        if not self.results:
            notes.append(
                "No falsifiers run. Tier is a claim, not a finding."
            )
        return tier, notes

    def report(self) -> str:
        tier, notes = self.tier()
        cp = self.counterparty
        lines = [
            "MECHANISM",
            f"  counterparty    {cp.agent.value} / {cp.compulsion.value}",
            f"  direction       {cp.direction}",
            f"  proxy           {cp.observable_proxy}",
            f"  calendar        {cp.calendar or '-'}",
            f"  barriers        {', '.join(b.value for b in self.persistence.barriers)}",
            f"  capacity        ${self.persistence.capacity_usd:,.0f}",
            f"  crisis state    {self.crisis_state.value}",
            "  falsifiers:",
        ]
        for r in self.results:
            lines.append(f"    [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
        lines.append(f"  --> EARNED TIER: {tier.value}")
        for n in notes:
            lines.append(f"      . {n}")
        return "\n".join(lines)

    def to_prior(self):
        """Convert to an EconomicPrior carrying the EARNED tier.

        Closes the loop: registry.register(prior=mech.to_prior()) means the
        trial multiplier applied by evaluate() reflects tested evidence, not
        a self-assigned label.
        """
        from .priors import EconomicPrior

        tier, notes = self.tier()
        cp = self.counterparty
        return EconomicPrior(
            tier=tier,
            counterparty=f"{cp.agent.value} under {cp.compulsion.value}: "
                         f"{cp.direction} [proxy: {cp.observable_proxy}]",
            persistence=f"barriers={','.join(b.value for b in self.persistence.barriers)}; "
                        f"capacity=${self.persistence.capacity_usd:,.0f}; "
                        f"{self.persistence.rationale}",
            crisis_state=self.crisis_state,
            falsifier="; ".join(f"{r.name}={'PASS' if r.passed else 'FAIL'}"
                                for r in self.results) or "none run",
        )
