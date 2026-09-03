"""Signal diagnostics that answer the question you actually asked.

"Does this signal improve vol-targeted returns?" decomposes into four separate
questions, and conflating them is how a shapeless blob becomes a conviction:

  1. Is the response MONOTONE in the signal? (binscatter, not raw scatter)
  2. Does it DECAY like a real signal? (IC across horizons)
  3. Does it survive COSTS? (breakeven cost per unit turnover)
  4. Is the equity curve distinguishable from luck? (null band)

Every function here takes strict care that vol scaling at time t uses only
information through t-1. Vol targeting is the most common source of subtle
lookahead in exactly this workflow, because the natural pandas idiom
(`returns.ewm().std()`) includes the current observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# vol targeting -- the leak-safe version
# ---------------------------------------------------------------------------

def ewm_vol(returns: np.ndarray, halflife: float = 20.0) -> np.ndarray:
    """EWM volatility using data strictly BEFORE each index.

    Element t is estimated from returns[:t] only. The first entries are NaN
    because you genuinely did not have an estimate yet -- do not backfill them,
    that is the leak.
    """
    r = np.asarray(returns, float)
    n = len(r)
    lam = 0.5 ** (1.0 / halflife)
    out = np.full(n, np.nan)
    var = np.nan
    for t in range(n):
        out[t] = np.sqrt(var) if np.isfinite(var) else np.nan
        x = r[t]
        if not np.isfinite(x):
            continue
        var = x * x if not np.isfinite(var) else lam * var + (1 - lam) * x * x
    return out


@dataclass(frozen=True)
class VolTargeted:
    net_returns: np.ndarray
    gross_returns: np.ndarray
    leverage: np.ndarray
    turnover: np.ndarray
    realised_vol: float


def vol_target(
    signal: np.ndarray,
    returns: np.ndarray,
    *,
    target_vol: float = 0.10,
    halflife: float = 20.0,
    periods_per_year: int = 252,
    max_leverage: float = 3.0,
    cost_per_turnover: float = 0.0,
) -> VolTargeted:
    """Position = clipped(target / forecast vol) * lagged signal.

    `signal` is used at t-1 to trade returns at t. Both the signal lag and the
    vol lag are enforced here rather than left to the caller, because that is
    where it always goes wrong.
    """
    s = np.asarray(signal, float)
    r = np.asarray(returns, float)
    if len(s) != len(r):
        raise ValueError(f"length mismatch: signal {len(s)}, returns {len(r)}")

    vol = ewm_vol(r, halflife) * np.sqrt(periods_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = np.clip(target_vol / vol, 0.0, max_leverage)

    pos = np.full(len(r), np.nan)
    pos[1:] = lev[1:] * s[:-1]            # signal known at t-1, vol through t-1
    pos = np.nan_to_num(pos, nan=0.0)

    gross = pos * r
    turn = np.abs(np.diff(pos, prepend=0.0))
    net = gross - cost_per_turnover * turn

    live = net[np.isfinite(net)]
    rv = float(live.std(ddof=1) * np.sqrt(periods_per_year)) if len(live) > 2 else np.nan
    return VolTargeted(net, gross, pos, turn, rv)


# ---------------------------------------------------------------------------
# 1. response shape -- binscatter, not a raw scatter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResponseCurve:
    bucket_mid: np.ndarray
    bucket_mean: np.ndarray
    bucket_lo: np.ndarray
    bucket_hi: np.ndarray
    bucket_n: np.ndarray
    monotonicity: float          # Spearman rho of bucket index vs bucket mean
    monotonicity_p: float

    def summary(self) -> str:
        lines = ["  bucket   n      mean ret        95% CI"]
        for i in range(len(self.bucket_mean)):
            lines.append(
                f"  {i+1:>4}   {int(self.bucket_n[i]):>5}  {self.bucket_mean[i]:+.5f}"
                f"   [{self.bucket_lo[i]:+.5f}, {self.bucket_hi[i]:+.5f}]"
            )
        lines.append(
            f"  monotonicity rho={self.monotonicity:+.3f} (p={self.monotonicity_p:.3f})"
        )
        return "\n".join(lines)


def signal_response(
    signal: np.ndarray,
    fwd_returns: np.ndarray,
    *,
    n_buckets: int = 8,
    n_boot: int = 2000,
    seed: int = 0,
) -> ResponseCurve:
    """Bucket the LAGGED signal, bootstrap the mean return in each bucket.

    Why not the scatter you suggested: a daily signal with a genuinely useful
    IC of 0.05 explains 0.25% of variance. The scatter is a shapeless cloud
    whether the signal works or not, so it cannot discriminate. What you want
    from the same data is whether the CONDITIONAL MEAN rises monotonically --
    and monotonicity across buckets is far stronger evidence than a slope,
    because it is much harder to produce by accident.
    """
    rng = np.random.default_rng(seed)
    s, r = np.asarray(signal, float), np.asarray(fwd_returns, float)
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]

    edges = np.quantile(s, np.linspace(0, 1, n_buckets + 1))
    edges[-1] += 1e-12
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_buckets - 1)

    mid = np.zeros(n_buckets); mean = np.zeros(n_buckets)
    lo = np.zeros(n_buckets); hi = np.zeros(n_buckets); cnt = np.zeros(n_buckets)
    for b in range(n_buckets):
        sel = r[idx == b]
        cnt[b] = len(sel)
        mid[b] = float(np.median(s[idx == b])) if len(sel) else np.nan
        if len(sel) < 3:
            mean[b] = lo[b] = hi[b] = np.nan
            continue
        mean[b] = float(sel.mean())
        draws = sel[rng.integers(0, len(sel), (n_boot, len(sel)))].mean(axis=1)
        lo[b], hi[b] = np.percentile(draws, [2.5, 97.5])

    good = np.isfinite(mean)
    rho, p = stats.spearmanr(np.arange(n_buckets)[good], mean[good])
    return ResponseCurve(mid, mean, lo, hi, cnt, float(rho), float(p))


# ---------------------------------------------------------------------------
# 2. decay profile
# ---------------------------------------------------------------------------

def decay_profile(
    signal: np.ndarray,
    returns: np.ndarray,
    horizons: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21),
) -> dict[int, float]:
    """Rank IC of the signal against forward returns at several horizons.

    Real signals decay smoothly -- information about tomorrow implies some
    information about the day after. An IC that is large at exactly one horizon
    and near zero either side is almost always an artifact of the construction,
    not an edge with a time constant.
    """
    s, r = np.asarray(signal, float), np.asarray(returns, float)
    out: dict[int, float] = {}
    for h in horizons:
        fwd = np.convolve(r, np.ones(h), mode="full")[h - 1:][: len(r)]
        a, b = s[:-h], fwd[h:]
        ok = np.isfinite(a) & np.isfinite(b)
        out[h] = float(stats.spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 20 else np.nan
    return out


# ---------------------------------------------------------------------------
# 3. breakeven cost
# ---------------------------------------------------------------------------

def breakeven_cost(vt: VolTargeted, *, bps: bool = True) -> float:
    """Cost per unit turnover at which the strategy earns exactly zero.

    The most useful single number in signal research. Compare it to your real
    spread + impact: if breakeven is 3bps and you cross a 5bp spread, no
    amount of statistical significance saves it.
    """
    g = vt.gross_returns[np.isfinite(vt.gross_returns)].sum()
    t = vt.turnover[np.isfinite(vt.turnover)].sum()
    if t <= 0:
        return float("inf")
    c = g / t
    return c * 1e4 if bps else c


# ---------------------------------------------------------------------------
# 4. null band -- the antidote to the cumprod trick
# ---------------------------------------------------------------------------

def equity_curve(pnl: np.ndarray) -> np.ndarray:
    """Correctly compounded. Note: cumprod(1 + r), NOT cumprod(r)."""
    r = np.nan_to_num(np.asarray(pnl, float))
    return np.cumprod(1.0 + r)


def null_band(
    pnl: np.ndarray,
    *,
    n_paths: int = 300,
    block: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-bootstrap equity curves under a zero-mean null.

    Returns (p5, p50, p95) envelopes. Plot your curve inside this band. An
    equity curve going "up and to the right" is the single most persuasive and
    least informative object in quant research -- the eye reads the cumulative
    ordering as trend even when the increments are noise. The band restores
    the comparison your intuition is silently skipping.
    """
    rng = np.random.default_rng(seed)
    r = np.nan_to_num(np.asarray(pnl, float))
    r = r - r.mean()                       # impose the null
    n = len(r)
    n_blocks = int(np.ceil(n / block))
    paths = np.empty((n_paths, n))
    for i in range(n_paths):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        paths[i] = np.cumprod(1.0 + r[idx])
    return (np.percentile(paths, 5, axis=0),
            np.percentile(paths, 50, axis=0),
            np.percentile(paths, 95, axis=0))


# ---------------------------------------------------------------------------
# 5. regression -- with the three corrections geom_smooth(method="lm") omits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalRegression:
    slope: float
    intercept: float
    r2: float
    t_ols: float
    t_nw: float
    nw_lags: int
    slope_winsorised: float
    slope_theilsen: float
    top1pct_share: float          # fraction of the slope from the top 1% by |x|
    n: int

    def summary(self) -> str:
        return "\n".join([
            f"  slope        {self.slope:+.6f}   R2 {self.r2:.4f}   n {self.n}",
            f"  t-stat       OLS {self.t_ols:+.2f}  ->  Newey-West "
            f"{self.t_nw:+.2f}  ({self.nw_lags} lags)",
            f"  robustness   winsorised {self.slope_winsorised:+.6f}   "
            f"Theil-Sen {self.slope_theilsen:+.6f}",
            f"  influence    top 1% of |signal| drives "
            f"{self.top1pct_share:.0%} of the slope",
        ])


def _newey_west_t(x: np.ndarray, y: np.ndarray, slope: float, intercept: float):
    """HAC standard error for the slope. Returns (t_stat, n_lags)."""
    n = len(x)
    lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    resid = y - (intercept + slope * x)
    xc = x - x.mean()
    h = xc * resid                       # score contributions
    s = float(h @ h)
    for j in range(1, lags + 1):
        g = float(h[j:] @ h[:-j])
        s += 2.0 * (1.0 - j / (lags + 1.0)) * g
    sxx = float(xc @ xc)
    if sxx <= 0 or s <= 0:
        return float("nan"), lags
    se = np.sqrt(s) / sxx
    return float(slope / se), lags


def signal_regression(
    signal: np.ndarray, fwd_returns: np.ndarray, *, winsor: float = 0.01
) -> SignalRegression:
    """Regress forward returns on the lagged signal, honestly.

    Three things `geom_smooth(method="lm")` will not tell you:

    1. Its confidence band assumes iid homoskedastic errors. Returns are
       neither, so the band is too narrow and the t-stat too large. Fixed here
       with Newey-West.
    2. OLS minimises squared error, so with excess kurtosis around 10 a handful
       of days determine the slope. `top1pct_share` measures exactly that.
    3. A positive slope is not a profitable strategy. Slope is return per unit
       of signal; PnL depends on your sizing function. Read this alongside
       breakeven_cost(), never instead of it.
    """
    x, y = np.asarray(signal, float), np.asarray(fwd_returns, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)

    slope, intercept, r, _, se = stats.linregress(x, y)
    t_ols = slope / se if se > 0 else np.nan
    t_nw, lags = _newey_west_t(x, y, slope, intercept)

    lo, hi = np.quantile(x, [winsor, 1 - winsor])
    ylo, yhi = np.quantile(y, [winsor, 1 - winsor])
    sw = stats.linregress(np.clip(x, lo, hi), np.clip(y, ylo, yhi)).slope

    sub = np.random.default_rng(0).choice(n, min(n, 600), replace=False)
    ts = float(stats.theilslopes(y[sub], x[sub])[0])

    cut = np.quantile(np.abs(x), 0.99)
    tail = np.abs(x) >= cut
    xc, yc = x - x.mean(), y - y.mean()
    num_all = float(xc @ yc)
    share = float(xc[tail] @ yc[tail]) / num_all if num_all != 0 else np.nan

    return SignalRegression(float(slope), float(intercept), float(r ** 2),
                            float(t_ols), t_nw, lags, float(sw), ts,
                            float(share), n)


# ---------------------------------------------------------------------------
# 6. drift decomposition -- is the "signal" just a disguised static position?
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftDecomposition:
    total: float
    static: float            # mean(signal) * mean(return): a constant position
    timing: float            # cov(signal, return): the actual information
    static_share: float
    timing_t_nw: float
    alpha_vs_underlying: float
    alpha_t_nw: float
    signal_mean: float

    def summary(self) -> str:
        verdict = ("DISGUISED STATIC POSITION" if self.static_share > 0.5
                   else "timing-driven")
        return "\n".join([
            f"  signal mean      {self.signal_mean:+.4f}"
            + ("   <-- non-zero: drift can masquerade as skill"
               if abs(self.signal_mean) > 0.1 else ""),
            f"  total E[pnl]     {self.total:+.6f}",
            f"    static  E[s]E[r]   {self.static:+.6f}"
            f"  ({self.static_share:+.0%} of total)",
            f"    timing  Cov(s,r)   {self.timing:+.6f}"
            f"   NW t = {self.timing_t_nw:+.2f}",
            f"  alpha vs underlying  {self.alpha_vs_underlying:+.6f}"
            f"   NW t = {self.alpha_t_nw:+.2f}",
            f"  --> {verdict}",
        ])


def decompose_drift(
    signal: np.ndarray, returns: np.ndarray
) -> DriftDecomposition:
    """Split E[signal x return] into a static leg and a timing leg.

        E[s_t-1 * r_t]  =  E[s]E[r]  +  Cov(s_t-1, r_t)
                           ^^^^^^^^     ^^^^^^^^^^^^^^^
                           buy and hold   actual forecasting

    The static leg is what a CONSTANT position of size mean(signal) would have
    earned from the asset's drift. It carries no information and will not
    survive a market that stops going up. If it dominates, you have not found
    a signal -- you have found a long position with extra steps.

    `alpha_vs_underlying` goes further: regress the strategy PnL on the
    underlying's own returns and report the intercept. That strips beta as
    well as drift.
    """
    s, r = np.asarray(signal, float), np.asarray(returns, float)
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]

    ms, mr = float(s.mean()), float(r.mean())
    static = ms * mr
    timing = float(((s - ms) * (r - mr)).mean())
    total = static + timing

    pnl = s * r
    t_timing, _ = _newey_west_t(s - ms, r, timing / max(float(((s - ms) ** 2).mean()), 1e-18), mr)

    fit = stats.linregress(r, pnl)
    t_alpha, _ = _newey_west_t(r, pnl - fit.slope * r,
                               0.0, fit.intercept)
    resid = pnl - (fit.intercept + fit.slope * r)
    n = len(resid)
    lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    sse = float(resid @ resid)
    for j in range(1, lags + 1):
        sse += 2.0 * (1 - j / (lags + 1.0)) * float(resid[j:] @ resid[:-j])
    se_alpha = np.sqrt(max(sse, 0.0) / n) / np.sqrt(n)
    t_alpha = float(fit.intercept / se_alpha) if se_alpha > 0 else float("nan")

    return DriftDecomposition(
        total, static, timing,
        static / total if total != 0 else float("nan"),
        float(t_timing), float(fit.intercept), t_alpha, ms,
    )
