"""A small, intuitive front door for EdgeLab.

The rigorous modules remain available directly.  ``Study`` simply removes the
ceremony for the common workflow:

    have an idea -> register it -> diagnose it -> evaluate it

A plain-English study defaults to an UNLABELED prior.  EdgeLab never guesses a
mechanism from prose.  If you later develop and test a real mechanism, register
it as a new hypothesis in the same family and a fresh OOS window; the old trial
history remains counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .diagnostics import (
    DriftDecomposition,
    ResponseCurve,
    SignalRegression,
    VolTargeted,
    breakeven_cost,
    decay_profile,
    decompose_drift,
    signal_regression,
    signal_response,
    vol_target,
)
from .evaluate import Evaluation, evaluate, redeflate
from .evidence import EvidenceRecord
from .mechanism import Mechanism
from .priors import CrisisState, EconomicPrior, PriorTier
from .registry import Hypothesis, Registry


def _slug(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "_" for c in text.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "study"


def unlabeled_prior(why: str | None = None) -> EconomicPrior:
    """Return the honest default prior for a study without a tested mechanism."""
    why_text = (why or "").strip()
    persistence = (
        f"Unverified idea at registration: {why_text}"
        if why_text
        else "No structural persistence mechanism identified at registration."
    )
    return EconomicPrior(
        tier=PriorTier.UNLABELED,
        counterparty="Counterparty not identified at registration; mechanism left unlabeled.",
        persistence=persistence,
        crisis_state=CrisisState.UNKNOWN,
        falsifier=(
            "The statistical claim must survive out-of-sample inference, costs, "
            "and multiplicity adjustment."
        ),
    )


@dataclass(frozen=True)
class StudyDiagnostics:
    """The default diagnostic bundle returned by :meth:`Study.diagnose`."""

    response: ResponseCurve
    regression: SignalRegression
    drift: DriftDecomposition
    decay: dict[int, float]
    portfolio: VolTargeted
    breakeven_cost_bps: float

    def summary(self) -> str:
        decay_text = ", ".join(
            f"{h}:{v:+.3f}" if np.isfinite(v) else f"{h}:nan"
            for h, v in self.decay.items()
        )
        return "\n".join([
            "STUDY DIAGNOSTICS",
            f"  response rho       {self.response.monotonicity:+.3f} "
            f"(p={self.response.monotonicity_p:.3f})",
            f"  regression NW t    {self.regression.t_nw:+.2f}",
            f"  drift static share {self.drift.static_share:+.0%}",
            f"  decay IC           {decay_text}",
            f"  realised vol       {self.portfolio.realised_vol:.3f}",
            f"  breakeven cost     {self.breakeven_cost_bps:.1f} bps/turnover",
        ])


class Study:
    """Friendly wrapper around a registered EdgeLab hypothesis.

    Use :func:`study` for the shortest path.  The underlying ``Registry`` and
    ``Hypothesis`` remain exposed so advanced users lose no control.
    """

    def __init__(
        self,
        registry: Registry,
        hypothesis: Hypothesis,
        *,
        idea: str = "",
        why: str = "",
        owns_registry: bool = False,
    ) -> None:
        self.registry = registry
        self.hypothesis = hypothesis
        self.idea = idea
        self.why = why
        self._owns_registry = owns_registry

    @classmethod
    def create(
        cls,
        *,
        registry: Registry,
        name: str,
        idea: str,
        universe: str,
        target: str,
        is_end: str,
        oos_start: str,
        family: str | None = None,
        feature_spec: str = "unversioned@v1",
        why: str | None = None,
        prior: EconomicPrior | None = None,
        notes: str = "",
    ) -> "Study":
        """Register a study from plain English.

        ``why`` is deliberately *not* used to infer a stronger prior.  Unless
        ``prior`` is explicitly supplied, the study is registered UNLABELED.
        """
        if not idea or len(idea.strip()) < 8:
            raise ValueError("idea must briefly state the trading claim you want to test.")
        if not universe or len(universe.strip()) < 3:
            raise ValueError("universe must say which market/assets are being tested.")
        if not target or len(target.strip()) < 3:
            raise ValueError("target must say what outcome the signal is meant to predict.")

        family = family or _slug(name)
        prior = prior or unlabeled_prior(why)
        plain_notes = [f"Idea: {idea.strip()}"]
        if why and why.strip():
            plain_notes.append(f"Why (unverified): {why.strip()}")
        if notes.strip():
            plain_notes.append(notes.strip())

        hyp = registry.register(
            family=family,
            name=name,
            prior=prior,
            estimand=target,
            universe=universe,
            feature_spec=feature_spec,
            is_end=is_end,
            oos_start=oos_start,
            notes="\n".join(plain_notes),
        )
        return cls(registry, hyp, idea=idea, why=why or "")

    @classmethod
    def load(cls, registry: Registry, hypothesis_id: str) -> "Study":
        hyp = registry.get(hypothesis_id)
        idea = ""
        why = ""
        for line in hyp.notes.splitlines():
            if line.startswith("Idea: "):
                idea = line[6:]
            elif line.startswith("Why (unverified): "):
                why = line[18:]
        return cls(registry, hyp, idea=idea, why=why)

    @property
    def id(self) -> str:
        return self.hypothesis.id

    @property
    def family(self) -> str:
        return self.hypothesis.family

    def summary(self) -> str:
        h = self.hypothesis
        story = self.why if self.why else "unknown (honestly left unlabeled)"
        return "\n".join([
            h.name.upper(),
            f"  id          {h.id}",
            f"  family      {h.family}",
            f"  idea        {self.idea or h.name}",
            f"  why         {story}",
            f"  universe    {h.universe}",
            f"  target      {h.estimand}",
            f"  feature     {h.feature_spec}",
            f"  split       IS through {h.is_end}; OOS starts {h.oos_start}",
            f"  prior       {h.prior.tier.value}",
            f"  trials      {self.registry.trial_count(h.family)}",
        ])

    def diagnose(
        self,
        signal: np.ndarray,
        returns: np.ndarray,
        *,
        target_vol: float = 0.10,
        halflife: float = 20.0,
        periods_per_year: int = 252,
        max_leverage: float = 3.0,
        cost_per_turnover: float = 0.0,
        n_buckets: int = 8,
        horizons: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21),
        n_boot: int = 2000,
        seed: int = 0,
    ) -> StudyDiagnostics:
        """Run the standard leak-aware diagnostic bundle in one call.

        ``signal[t]`` is treated as information known at t.  Response,
        regression and drift therefore compare ``signal[t]`` with
        ``returns[t+1]``.  ``vol_target`` enforces the same one-period lag.
        """
        s = np.asarray(signal, float)
        r = np.asarray(returns, float)
        if len(s) != len(r):
            raise ValueError(f"length mismatch: signal {len(s)}, returns {len(r)}")
        if len(s) < 25:
            raise ValueError("Need at least 25 observations for diagnostics.")

        # Explicit one-period alignment for diagnostics whose low-level API
        # expects already-aligned signal and forward return arrays.
        response = signal_response(
            s[:-1], r[1:], n_buckets=n_buckets, n_boot=n_boot, seed=seed
        )
        regression = signal_regression(s[:-1], r[1:])
        drift = decompose_drift(s[:-1], r[1:])
        decay = decay_profile(s, r, horizons=horizons)
        portfolio = vol_target(
            s,
            r,
            target_vol=target_vol,
            halflife=halflife,
            periods_per_year=periods_per_year,
            max_leverage=max_leverage,
            cost_per_turnover=cost_per_turnover,
        )
        return StudyDiagnostics(
            response=response,
            regression=regression,
            drift=drift,
            decay=decay,
            portfolio=portfolio,
            breakeven_cost_bps=breakeven_cost(portfolio),
        )

    def test(
        self,
        pnl: np.ndarray,
        *,
        variant: str = "v1",
        window: str = "oos",
        periods_per_year: int = 252,
        n_boot: int = 5000,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> Evaluation:
        """Evaluate net-of-cost PnL and automatically log the trial."""
        return evaluate(
            self.registry,
            self.id,
            pnl,
            variant=variant,
            window=window,
            periods_per_year=periods_per_year,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )

    def redeflate(self, evaluation: Evaluation) -> Evaluation:
        """Score an earlier winner against the family's complete trial history."""
        return redeflate(self.registry, evaluation)

    def evidence(self, evaluation: Evaluation) -> EvidenceRecord:
        """Freeze a promotion-grade validation artifact for this study.

        If more variants were tested after ``evaluation`` ran, EdgeLab first
        re-deflates it against the family's complete trial history.  Exported
        evidence therefore cannot accidentally preserve an optimistic
        multiplicity count from the middle of a sweep.
        """
        if evaluation.hypothesis_id != self.id:
            raise ValueError(
                f"evaluation belongs to {evaluation.hypothesis_id!r}, not study {self.id!r}"
            )
        final = evaluation
        current_trials = self.registry.trial_count(self.family)
        if current_trials > evaluation.raw_trials:
            final = self.redeflate(evaluation)
        return EvidenceRecord.from_evaluation(self.hypothesis, final)

    def write_evidence(
        self, evaluation: Evaluation, path: str | Path
    ) -> EvidenceRecord:
        """Write validation evidence for Conductor/other orchestrators."""
        record = self.evidence(evaluation)
        record.write(path)
        return record

    def with_mechanism(
        self,
        mechanism: Mechanism,
        *,
        is_end: str,
        oos_start: str,
        name: str | None = None,
        feature_spec: str | None = None,
        notes: str = "",
    ) -> "Study":
        """Register a mechanism-backed revision using a *fresh* OOS split.

        This intentionally creates a new hypothesis in the same family.  A
        story developed after seeing results is another research decision; it
        must not retroactively upgrade the old test.  Family trial counts are
        therefore preserved.
        """
        h = self.hypothesis
        extra = (
            f"Mechanism-backed revision of hypothesis {h.id}. "
            "Prior trial history remains in the same family."
        )
        if notes.strip():
            extra += "\n" + notes.strip()
        revised = self.registry.register(
            family=h.family,
            name=name or f"{h.name} — mechanism revision",
            prior=mechanism.to_prior(),
            estimand=h.estimand,
            universe=h.universe,
            feature_spec=feature_spec or h.feature_spec,
            is_end=is_end,
            oos_start=oos_start,
            notes="\n".join(filter(None, [f"Idea: {self.idea}", extra])),
        )
        return Study(self.registry, revised, idea=self.idea)

    def close(self) -> None:
        if self._owns_registry:
            self.registry.close()

    def __enter__(self) -> "Study":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def study(
    *,
    name: str,
    idea: str,
    universe: str,
    target: str,
    is_end: str,
    oos_start: str,
    why: str | None = None,
    family: str | None = None,
    feature_spec: str = "unversioned@v1",
    db: str | Path = "edgelab.db",
    registry: Registry | None = None,
    prior: EconomicPrior | None = None,
    notes: str = "",
) -> Study:
    """Shortest path from a trading idea to a registered EdgeLab study.

    Example::

        s = edgelab.study(
            name="5d reversal",
            idea="Large five-day losers rebound over the next five days",
            universe="US large caps",
            target="mean next-5d net portfolio PnL",
            is_end="2024-12-31",
            oos_start="2025-01-01",
        )

    If ``registry`` is omitted, ``db`` is opened and owned by the returned
    study.  Call ``s.close()`` or use it as a context manager.
    """
    owns = registry is None
    reg = registry or Registry(db)
    out = Study.create(
        registry=reg,
        name=name,
        idea=idea,
        why=why,
        universe=universe,
        target=target,
        family=family,
        feature_spec=feature_spec,
        is_end=is_end,
        oos_start=oos_start,
        prior=prior,
        notes=notes,
    )
    out._owns_registry = owns
    return out
