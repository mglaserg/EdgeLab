"""Evaluation that cannot flatter you.

Design constraints, in order of importance:

1. It refuses to produce a number for an unregistered hypothesis.
2. The returned object structurally contains the trial count. There is no
   code path that reports a Sharpe without also reporting how many variants
   were searched to find it.
3. Inference is on the *mean of net PnL*, via a stationary bootstrap that
   preserves autocorrelation -- not a distributional test. Two return series
   can differ decisively in shape and have identical means; you cannot spend
   a shape difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .priors import PriorTier
from .registry import Registry

_EULER = 0.5772156649015329


# ---------------------------------------------------------------------------
# block length selection
# ---------------------------------------------------------------------------

def optimal_block_length(x: np.ndarray) -> float:
    """Mean block length for the stationary bootstrap.

    Practical Politis-White flavour: scale N^(1/3) by the persistence implied
    by first-order autocorrelation. Serially correlated PnL needs longer
    blocks, otherwise the bootstrap treats dependent observations as
    independent and hands back a confidence interval that is far too tight.
    """
    n = len(x)
    if n < 8:
        return 1.0
    x = x - x.mean()
    denom = float(x @ x)
    rho = float(x[:-1] @ x[1:]) / denom if denom > 0 else 0.0
    rho = float(np.clip(rho, -0.95, 0.95))
    base = n ** (1.0 / 3.0)
    if abs(rho) < 1e-6:
        return max(1.0, base)
    persistence = (2.0 * rho ** 2 / (1.0 - rho ** 2) ** 2) ** (1.0 / 3.0)
    return float(np.clip(base * max(persistence, 0.5), 1.0, n / 4.0))


def stationary_bootstrap_indices(
    n: int, block_len: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano: geometric block lengths, circular wrap.

    Geometric blocks keep the resampled series stationary, which fixed-length
    blocks do not.
    """
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty(n, dtype=np.int64)
    i = int(rng.integers(n))
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(n))
        else:
            i = (i + 1) % n
    return idx


# ---------------------------------------------------------------------------
# deflated sharpe
# ---------------------------------------------------------------------------

def expected_max_sharpe(n_trials: float, sharpe_variance: float) -> float:
    """Expected maximum Sharpe under the null that every trial has zero edge.

    This is the bar your best variant must clear. It rises with the number of
    variants searched, which is exactly why the trial count must be honest.
    """
    k = max(float(n_trials), 2.0)
    sd = math.sqrt(max(sharpe_variance, 1e-12))
    z1 = stats.norm.ppf(1.0 - 1.0 / k)
    z2 = stats.norm.ppf(1.0 - 1.0 / (k * math.e))
    return sd * ((1.0 - _EULER) * z1 + _EULER * z2)


def deflated_sharpe(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: float,
    sharpe_variance: float,
) -> tuple[float, float]:
    """Bailey / Lopez de Prado DSR. Returns (probability, hurdle).

    All Sharpes are per-observation, NOT annualised -- annualising before
    deflating silently inflates the result. Negative skew and fat tails, which
    is what a short-strangle PnL series looks like, raise the bar.
    """
    if n_obs < 3:
        return float("nan"), float("nan")
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    denom_sq = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    if denom_sq <= 0:
        return float("nan"), sr0
    z = (sharpe - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z)), float(sr0)


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evaluation:
    """Every field here is mandatory. The trial count is not optional context."""

    hypothesis_id: str
    family: str
    variant: str
    window: str
    tier: PriorTier

    n_obs: int
    mean_pnl: float
    sharpe_per_obs: float
    sharpe_annualised: float
    skew: float
    excess_kurtosis: float

    mean_ci: tuple[float, float]
    sharpe_ci: tuple[float, float]
    block_length: float

    raw_trials: int
    effective_trials: float
    sharpe_hurdle: float
    dsr: float

    warnings: list[str] = field(default_factory=list)
    trial_id: str = ""

    @property
    def survives(self) -> bool:
        """DSR above 0.95 and a mean CI that excludes zero. Both, not either."""
        return self.dsr > 0.95 and self.mean_ci[0] > 0.0

    def summary(self) -> str:
        verdict = "SURVIVES" if self.survives else "does not clear the bar"
        lines = [
            f"{self.family} / {self.variant}  [{self.window}]",
            f"  tier            {self.tier.value}",
            f"  n_obs           {self.n_obs}",
            f"  Sharpe (ann.)   {self.sharpe_annualised:+.3f}"
            f"   CI [{self.sharpe_ci[0]:+.3f}, {self.sharpe_ci[1]:+.3f}]",
            f"  mean PnL        {self.mean_pnl:+.6f}"
            f"   CI [{self.mean_ci[0]:+.6f}, {self.mean_ci[1]:+.6f}]",
            f"  skew / exkurt   {self.skew:+.2f} / {self.excess_kurtosis:+.2f}",
            f"  block length    {self.block_length:.1f}",
            f"  trials          {self.raw_trials} raw"
            f" -> {self.effective_trials:.0f} effective (tier haircut)",
            f"  hurdle Sharpe   {self.sharpe_hurdle:.4f} per obs"
            f"  (realised {self.sharpe_per_obs:.4f})",
            f"  DSR             {self.dsr:.3f}   --> {verdict}",
        ]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def redeflate(registry: Registry, ev: Evaluation) -> Evaluation:
    """Re-deflate a result against the family's FINAL trial count.

    Necessary because `evaluate` counts trials as-of the run. If you sweep
    forty variants and keep the best, the winner was scored against however
    many trials existed when it happened to run -- not forty. Call this on the
    survivor after a sweep completes, or you understate your own search.
    """
    hyp = registry.get(ev.hypothesis_id)
    raw = registry.trial_count(hyp.family)
    effective = raw * hyp.prior.trial_multiplier
    history = registry.family_sharpes(hyp.family)
    sr_var = (float(np.var(history, ddof=1)) if len(history) >= 5
              else 1.0 / (ev.n_obs - 1))
    dsr, hurdle = deflated_sharpe(
        ev.sharpe_per_obs, ev.n_obs, ev.skew, ev.excess_kurtosis,
        effective, sr_var,
    )
    notes = list(ev.warnings) + [
        f"Re-deflated against the full family history ({raw} trials), not the "
        f"{ev.raw_trials} that existed when this variant ran."
    ]
    return Evaluation(
        **{**ev.__dict__,
           "raw_trials": raw,
           "effective_trials": effective,
           "sharpe_hurdle": hurdle,
           "dsr": dsr,
           "warnings": notes}
    )


def evaluate(
    registry: Registry,
    hypothesis_id: str,
    pnl: np.ndarray,
    *,
    variant: str,
    window: str,
    periods_per_year: int = 252,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Evaluation:
    """Evaluate a net-of-cost PnL series against a pre-registered hypothesis.

    `pnl` must already be net of transaction costs and financing. Evaluating
    gross PnL is the most common way a backtest survives that should not.
    """
    hyp = registry.get(hypothesis_id)          # raises if unregistered
    x = np.asarray(pnl, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")

    warnings: list[str] = []

    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    sr = mu / sd if sd > 0 else 0.0
    skew = float(stats.skew(x))
    exkurt = float(stats.kurtosis(x))          # excess

    block = optimal_block_length(x)
    rng = np.random.default_rng(seed)
    boot_mu = np.empty(n_boot)
    boot_sr = np.empty(n_boot)
    for b in range(n_boot):
        s = x[stationary_bootstrap_indices(n, block, rng)]
        m = s.mean()
        v = s.std(ddof=1)
        boot_mu[b] = m
        boot_sr[b] = (m / v) if v > 0 else 0.0

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    mean_ci = (float(np.percentile(boot_mu, lo_q)),
               float(np.percentile(boot_mu, hi_q)))
    sr_ci = (float(np.percentile(boot_sr, lo_q)) * math.sqrt(periods_per_year),
             float(np.percentile(boot_sr, hi_q)) * math.sqrt(periods_per_year))

    # Trial accounting. Count THIS run, so the current attempt is included.
    raw_trials = registry.trial_count(hyp.family) + 1
    effective_trials = raw_trials * hyp.prior.trial_multiplier

    # Variance of trial Sharpes: use the family's own history once it is long
    # enough, otherwise fall back to the asymptotic variance of a zero-edge
    # Sharpe estimate, 1/(n-1).
    history = registry.family_sharpes(hyp.family)
    if len(history) >= 5:
        sr_var = float(np.var(history, ddof=1))
    else:
        sr_var = 1.0 / (n - 1)
        warnings.append(
            "Fewer than 5 logged trials in this family; DSR uses the "
            "asymptotic Sharpe variance, which is usually optimistic."
        )

    dsr, hurdle = deflated_sharpe(sr, n, skew, exkurt, effective_trials, sr_var)

    if hyp.prior.tier is PriorTier.UNLABELED:
        warnings.append(
            "UNLABELED prior: no counterparty story. Charged a 20x trial "
            "multiplier. Find the flow or expect this not to replicate."
        )
    if skew < -0.5 and sr > 0:
        warnings.append(
            f"Negative skew ({skew:.2f}) with positive Sharpe -- the classic "
            f"short-optionality signature. Check the loss tail is real, not a "
            f"sample in which the bad state never occurred."
        )
    if block > n / 10:
        warnings.append(
            f"Block length {block:.0f} vs n={n}: strong persistence means "
            f"your effective sample size is far smaller than the raw count."
        )
    if window.lower().startswith("is") or "in-sample" in window.lower():
        warnings.append("In-sample window. This is a diagnostic, not evidence.")

    trial_id = registry.log_trial(
        hypothesis_id=hypothesis_id,
        variant=variant,
        sharpe=sr,
        n_obs=n,
        window=window,
        payload={"mean": mu, "skew": skew, "exkurt": exkurt, "block": block},
    )

    return Evaluation(
        hypothesis_id=hypothesis_id,
        family=hyp.family,
        variant=variant,
        window=window,
        tier=hyp.prior.tier,
        n_obs=n,
        mean_pnl=mu,
        sharpe_per_obs=sr,
        sharpe_annualised=sr * math.sqrt(periods_per_year),
        skew=skew,
        excess_kurtosis=exkurt,
        mean_ci=mean_ci,
        sharpe_ci=sr_ci,
        block_length=block,
        raw_trials=raw_trials,
        effective_trials=effective_trials,
        sharpe_hurdle=hurdle,
        dsr=dsr,
        warnings=warnings,
        trial_id=trial_id,
    )
