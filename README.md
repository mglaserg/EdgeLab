# edgelab

A referee, not a search engine. It cannot find edges for you; it can stop you
believing false ones. Expect your apparent hit rate to get *worse* — that is
the tool working.

## Layout

| module | job |
|---|---|
| `priors.py` | tier taxonomy (`RISK_PREMIUM` → `UNLABELED`) and trial multipliers |
| `mechanism.py` | typed counterparty + persistence; falsifiers that **cap** the asserted tier |
| `registry.py` | SQLite pre-registration, append-only trial log |
| `evaluate.py` | stationary bootstrap CIs, deflated Sharpe fed by the real trial count |
| `diagnostics.py` | leak-safe vol targeting, binscatter response, decay, breakeven cost, null band, drift decomposition |

## Intended workflow

1. Write the `Mechanism` **before** looking at PnL. Name the agent, the
   compulsion, and an *observable proxy series*. The falsifiers are your
   predictions; written afterwards they are just descriptions.
2. `registry.register(prior=mech.to_prior())` — the tier is earned, not asserted.
3. Run `diagnostics` on the signal. Check `decompose_drift` first: if the
   static leg dominates, it is a buy-and-hold with extra steps.
4. `evaluate(...)`. After a sweep completes, `redeflate(...)` against the
   family's full trial count.

## Known gaps

- Feature versioning is a string. Hash your upstream config into it, or
  refitting a surface silently invalidates prior trials.
- Revising a mechanism after seeing a falsifier fail is a searched hypothesis
  and nothing counts it.
- No PIT panel layer yet. Nothing guarantees a feature was available at the
  decision timestamp — the usual killer in options data.

## Demos

    python demo.py              # a mined winner dies; a real edge is marginal
    python demo_mechanism.py    # identical stories, only one survives its falsifier
    python demo_diagnostics.py  # binscatter, decay, breakeven cost, null band
