"""Portable validation evidence for orchestration systems such as Conductor.

EdgeLab owns statistical validation.  It should not own promotion or execution.
This module turns a completed EdgeLab evaluation into a small, durable JSON
artifact that another process can inspect without importing EdgeLab internals.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .evaluate import Evaluation
from .registry import Hypothesis

EVIDENCE_SCHEMA_VERSION = "edgelab.validation.v1"
PRODUCER = "edgelab"
ARTIFACT_TYPE = "validation"
PRODUCER_VERSION = __version__

_OOS_WINDOWS = {
    "oos",
    "out-of-sample",
    "out_of_sample",
    "holdout",
    "sealed-holdout",
    "sealed_holdout",
    "validation",
}


def _json_safe(value: Any) -> Any:
    """Convert numpy-ish / non-finite values into strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy scalar types expose item(); avoid importing numpy in this module.
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return value


def _normalise_window(window: str) -> str:
    return window.strip().lower().replace(" ", "-")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Self-contained EdgeLab validation result suitable for durable storage."""

    evidence_id: str
    schema_version: str
    producer: str
    artifact_type: str
    producer_version: str
    decision: str
    eligible_for_promotion: bool
    hypothesis: dict[str, Any]
    evaluation: dict[str, Any]

    @classmethod
    def from_evaluation(
        cls,
        hypothesis: Hypothesis,
        evaluation: Evaluation,
    ) -> "EvidenceRecord":
        if evaluation.hypothesis_id != hypothesis.id:
            raise ValueError(
                "evaluation does not belong to hypothesis "
                f"{hypothesis.id!r}: got {evaluation.hypothesis_id!r}"
            )

        explicit_oos = _normalise_window(evaluation.window) in _OOS_WINDOWS
        eligible = bool(explicit_oos and evaluation.survives)
        decision = "pass" if eligible else ("fail" if explicit_oos else "diagnostic")

        hypothesis_payload = {
            "id": hypothesis.id,
            "family": hypothesis.family,
            "name": hypothesis.name,
            "tier": hypothesis.prior.tier.value,
            "counterparty": hypothesis.prior.counterparty,
            "persistence": hypothesis.prior.persistence,
            "crisis_state": hypothesis.prior.crisis_state.value,
            "falsifier": hypothesis.prior.falsifier,
            "estimand": hypothesis.estimand,
            "universe": hypothesis.universe,
            "feature_spec": hypothesis.feature_spec,
            "is_end": hypothesis.is_end,
            "oos_start": hypothesis.oos_start,
            "registered_at": hypothesis.registered_at,
            "notes": hypothesis.notes,
        }
        evaluation_payload = {
            "trial_id": evaluation.trial_id or None,
            "variant": evaluation.variant,
            "window": evaluation.window,
            "n_obs": evaluation.n_obs,
            "mean_pnl": evaluation.mean_pnl,
            "mean_ci": list(evaluation.mean_ci),
            "sharpe_per_obs": evaluation.sharpe_per_obs,
            "sharpe_annualised": evaluation.sharpe_annualised,
            "sharpe_ci": list(evaluation.sharpe_ci),
            "skew": evaluation.skew,
            "excess_kurtosis": evaluation.excess_kurtosis,
            "block_length": evaluation.block_length,
            "raw_trials": evaluation.raw_trials,
            "effective_trials": evaluation.effective_trials,
            "sharpe_hurdle": evaluation.sharpe_hurdle,
            "dsr": evaluation.dsr,
            "survives": evaluation.survives,
            "warnings": list(evaluation.warnings),
        }

        body = _json_safe(
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "producer": PRODUCER,
                "artifact_type": ARTIFACT_TYPE,
                "producer_version": PRODUCER_VERSION,
                "decision": decision,
                "eligible_for_promotion": eligible,
                "hypothesis": hypothesis_payload,
                "evaluation": evaluation_payload,
            }
        )
        canonical = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()[:24]
        evidence_id = f"edgelab:{hypothesis.id}:{digest}"
        return cls(evidence_id=evidence_id, **body)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "schema_version": self.schema_version,
            "producer": self.producer,
            "artifact_type": self.artifact_type,
            "producer_version": self.producer_version,
            "decision": self.decision,
            "eligible_for_promotion": self.eligible_for_promotion,
            "hypothesis": self.hypothesis,
            "evaluation": self.evaluation,
        }

    def write(self, path: str | Path) -> Path:
        """Write strict, deterministic JSON and return the resolved artifact path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return out.resolve()

    def conductor_reference(self, location: str | Path) -> dict[str, str]:
        """Return the exact fields consumed by Conductor's EvidenceReference."""
        return {
            "producer": self.producer,
            "artifact_type": self.artifact_type,
            "location": str(location),
            "version": self.producer_version,
        }
