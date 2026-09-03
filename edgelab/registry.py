"""Hypothesis registry.

Two jobs:

1. Force pre-registration. A hypothesis records its economic prior, estimand
   and OOS window *before* any result exists. Evaluation refuses to run
   against an unregistered hypothesis.

2. Count trials honestly. Every evaluation -- including the ones you did not
   like -- is appended to an immutable trial log, keyed by *family*. The
   deflated Sharpe ratio then consumes the real number of variants you tried,
   not the number you remember trying.

The registry is deliberately append-only for trials. There is no delete API,
because the failure mode this exists to prevent is you quietly dropping the
thirty-nine variants that did not work.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .priors import CrisisState, EconomicPrior, PriorTier

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id                TEXT PRIMARY KEY,
    family            TEXT NOT NULL,
    name              TEXT NOT NULL,
    tier              TEXT NOT NULL,
    counterparty      TEXT NOT NULL,
    persistence       TEXT NOT NULL,
    crisis_state      TEXT NOT NULL,
    falsifier         TEXT NOT NULL,
    estimand          TEXT NOT NULL,
    universe          TEXT NOT NULL,
    feature_spec      TEXT NOT NULL,
    is_end            TEXT NOT NULL,
    oos_start         TEXT NOT NULL,
    registered_at     TEXT NOT NULL,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    trial_id       TEXT PRIMARY KEY,
    hypothesis_id  TEXT NOT NULL,
    family         TEXT NOT NULL,
    run_at         TEXT NOT NULL,
    variant        TEXT NOT NULL,
    sharpe         REAL,
    n_obs          INTEGER,
    window         TEXT NOT NULL,
    payload        TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE INDEX IF NOT EXISTS idx_trials_family ON trials(family);
CREATE INDEX IF NOT EXISTS idx_trials_hyp ON trials(hypothesis_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Hypothesis:
    """A pre-registered claim. Immutable once written."""

    id: str
    family: str            # Variants of the same idea share a family.
    name: str
    prior: EconomicPrior
    estimand: str          # "mean daily net PnL of the spread portfolio"
    universe: str          # "SPY weekly options, 2016-2024, liquidity filtered"
    feature_spec: str      # versioned identifier of the feature computation
    is_end: str            # last date you are allowed to look at, ISO
    oos_start: str         # first OOS date, ISO
    registered_at: str = field(default_factory=_now)
    notes: str = ""


class UnregisteredHypothesis(RuntimeError):
    """Raised when evaluation is attempted without pre-registration."""


class Registry:
    """SQLite-backed. One file per research program."""

    def __init__(self, path: str | Path = "edgelab.db") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- registration -----------------------------------------------------

    def register(
        self,
        *,
        family: str,
        name: str,
        prior: EconomicPrior,
        estimand: str,
        universe: str,
        feature_spec: str,
        is_end: str,
        oos_start: str,
        notes: str = "",
    ) -> Hypothesis:
        """Write a hypothesis before you have seen a result for it."""
        if oos_start <= is_end:
            raise ValueError(
                f"oos_start ({oos_start}) must be strictly after is_end "
                f"({is_end}). Overlapping windows are not an out-of-sample test."
            )
        hyp = Hypothesis(
            id=uuid.uuid4().hex[:12],
            family=family,
            name=name,
            prior=prior,
            estimand=estimand,
            universe=universe,
            feature_spec=feature_spec,
            is_end=is_end,
            oos_start=oos_start,
            notes=notes,
        )
        self._conn.execute(
            "INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                hyp.id, hyp.family, hyp.name, prior.tier.value,
                prior.counterparty, prior.persistence,
                prior.crisis_state.value, prior.falsifier,
                hyp.estimand, hyp.universe, hyp.feature_spec,
                hyp.is_end, hyp.oos_start, hyp.registered_at, hyp.notes,
            ),
        )
        self._conn.commit()
        return hyp

    def get(self, hypothesis_id: str) -> Hypothesis:
        row = self._conn.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
        ).fetchone()
        if row is None:
            raise UnregisteredHypothesis(
                f"No hypothesis {hypothesis_id!r}. Register the claim -- prior, "
                f"estimand, universe, OOS window -- before evaluating it."
            )
        return Hypothesis(
            id=row["id"],
            family=row["family"],
            name=row["name"],
            prior=EconomicPrior(
                tier=PriorTier(row["tier"]),
                counterparty=row["counterparty"],
                persistence=row["persistence"],
                crisis_state=CrisisState(row["crisis_state"]),
                falsifier=row["falsifier"],
            ),
            estimand=row["estimand"],
            universe=row["universe"],
            feature_spec=row["feature_spec"],
            is_end=row["is_end"],
            oos_start=row["oos_start"],
            registered_at=row["registered_at"],
            notes=row["notes"] or "",
        )

    # -- trials -----------------------------------------------------------

    def log_trial(
        self,
        *,
        hypothesis_id: str,
        variant: str,
        sharpe: float | None,
        n_obs: int,
        window: str,
        payload: dict | None = None,
    ) -> str:
        """Append-only. Called automatically by evaluate(); do not skip it."""
        hyp = self.get(hypothesis_id)
        trial_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?)",
            (
                trial_id, hypothesis_id, hyp.family, _now(), variant,
                sharpe, n_obs, window, json.dumps(payload or {}),
            ),
        )
        self._conn.commit()
        return trial_id

    def trial_count(self, family: str) -> int:
        """How many variants of this idea have actually been run."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM trials WHERE family = ?", (family,)
        ).fetchone()
        return int(row["n"])

    def family_sharpes(self, family: str) -> list[float]:
        """All Sharpes ever produced in a family -- feeds the DSR variance."""
        rows = self._conn.execute(
            "SELECT sharpe FROM trials WHERE family = ? AND sharpe IS NOT NULL",
            (family,),
        ).fetchall()
        return [float(r["sharpe"]) for r in rows]

    def crisis_exposure(self) -> dict[str, int]:
        """Portfolio-level check: how concentrated is your book's bad state?

        If one crisis state dominates, your strategies are not as uncorrelated
        as their sample correlation matrix suggests.
        """
        rows = self._conn.execute(
            "SELECT crisis_state, COUNT(*) AS n FROM hypotheses GROUP BY crisis_state"
        ).fetchall()
        return {r["crisis_state"]: int(r["n"]) for r in rows}

    def to_dict(self, hypothesis_id: str) -> dict:
        h = self.get(hypothesis_id)
        d = asdict(h)
        d["prior"]["tier"] = h.prior.tier.value
        d["prior"]["crisis_state"] = h.prior.crisis_state.value
        return d

    def close(self) -> None:
        self._conn.close()
