"""Typed economic priors.

A hypothesis is not admissible until you can name the counterparty and say why
their flow persists. The tier controls how harsh the multiplicity haircut is:
strong structural priors start with real prior mass, unlabeled statistical
regularities do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PriorTier(str, Enum):
    """Why does this edge exist? Ordered from strongest to weakest prior."""

    RISK_PREMIUM = "risk_premium"
    # You are paid to hold something others will not. VRP, term premium, carry.
    # Falsifier: the strategy MUST lose money when the named risk realizes.
    # A "risk premium" with no bad state is mislabeled.

    CONSTRAINT = "constraint"
    # Someone must trade regardless of price. Index rebalance, month-end
    # pension flows, expiry pin, forced deleveraging, mandate limits.
    # Falsifier: name the mandate and its calendar. Effect should be absent
    # off-calendar.

    FRICTION = "friction"
    # Segmentation, capital charges, borrow limits, balance-sheet costs.
    # Falsifier: explain why arbitrage capital has not closed it, and what
    # would happen if the friction were removed.

    BEHAVIORAL = "behavioral"
    # Underreaction, anchoring, disposition. Weakest tier: "investors are
    # irrational" is unfalsifiable as stated, so this pays the largest haircut.

    UNLABELED = "unlabeled"
    # An honest admission: a statistical regularity with no story. Testable,
    # but at the punitive tier. Do not let this become the default.


#: Multiplies the effective trial count used by the deflated Sharpe ratio.
#: Weak priors are charged as if you had searched a much larger space, because
#: in effect you did -- the space of stories you could have told after the fact.
TIER_TRIAL_MULTIPLIER: dict[PriorTier, float] = {
    PriorTier.RISK_PREMIUM: 1.0,
    PriorTier.CONSTRAINT: 1.5,
    PriorTier.FRICTION: 2.0,
    PriorTier.BEHAVIORAL: 5.0,
    PriorTier.UNLABELED: 20.0,
}


class CrisisState(str, Enum):
    """Which regime does this strategy lose in?

    The point of this field is portfolio-level, not strategy-level. If every
    accepted hypothesis answers EQUITY_CRASH, the measured diversification of
    a multi-strategy book is an artifact of a calm sample.
    """

    EQUITY_CRASH = "equity_crash"
    VOL_SPIKE = "vol_spike"
    LIQUIDITY_SPIRAL = "liquidity_spiral"
    RATES_SHOCK = "rates_shock"
    CARRY_UNWIND = "carry_unwind"
    CROWDING_UNWIND = "crowding_unwind"
    IDIOSYNCRATIC = "idiosyncratic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EconomicPrior:
    """The mandatory narrative fields, typed so they cannot be hand-waved."""

    tier: PriorTier
    counterparty: str        # Who is on the other side of this trade?
    persistence: str         # Why do they keep showing up? Why isn't it arbed?
    crisis_state: CrisisState
    falsifier: str           # What observation would kill this hypothesis?

    def __post_init__(self) -> None:
        for field_name in ("counterparty", "persistence", "falsifier"):
            value = getattr(self, field_name)
            if not value or len(value.strip()) < 15:
                raise ValueError(
                    f"EconomicPrior.{field_name} must be a real answer, not a "
                    f"placeholder (got {value!r}). If you cannot fill this in, "
                    f"register the hypothesis as PriorTier.UNLABELED and accept "
                    f"the haircut."
                )
        if self.tier is PriorTier.RISK_PREMIUM and self.crisis_state is CrisisState.UNKNOWN:
            raise ValueError(
                "A risk premium with no identified bad state is mislabeled. "
                "Name the crisis state, or downgrade the tier."
            )

    @property
    def trial_multiplier(self) -> float:
        return TIER_TRIAL_MULTIPLIER[self.tier]
