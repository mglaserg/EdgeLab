# edgelab

**A referee, not a search engine.** EdgeLab cannot find edges for you; it can
stop you believing false ones. Expect your apparent hit rate to get *worse* —
that is the tool working.

## Start here: test an idea

You do **not** need a complete economic mechanism to start using EdgeLab.
If you have a trading idea but do not yet know why it exists, register it
honestly as an unlabeled study:

```python
import edgelab

s = edgelab.study(
    name="5d reversal",
    idea="Stocks with unusually bad 5-day returns subsequently rebound.",
    why="Possibly forced selling; not established yet.",
    universe="US large caps",
    target="mean next-5d net portfolio PnL",
    feature_spec="reversal_5d@v1",
    is_end="2024-12-31",
    oos_start="2025-01-01",
)

print(s.summary())

d = s.diagnose(signal, returns, cost_per_turnover=0.0005)
print(d.summary())

ev = s.test(d.portfolio.net_returns, variant="baseline")
print(ev.summary())

s.close()
```

That is the normal workflow:

> **Have an idea → register it → inspect it → test it.**

The simple interface deliberately defaults to `UNLABELED`. EdgeLab does not
read your prose and magically promote a story to a stronger economic prior.
An unlabeled study pays the 20x multiplicity haircut until you actually earn a
better mechanism.

## Guided CLI

After installing the package:

```text
edgelab new
```

EdgeLab asks only the questions needed to preregister the test:

```text
What are you investigating?
> Very high VIX1D seems to predict falling volatility.

Why might it happen? (press Enter if you do not know)
>

That's okay. EdgeLab will register it as UNLABELED rather than invent a story.
```

To inspect a registered study later:

```text
edgelab show <hypothesis_id>
```

Use a different registry file with `--db`, for example:

```text
edgelab --db data/futurescope_edgelab.db new
```

Parent directories are created automatically.

## When to use a mechanism

A mechanism is an **advanced evidence layer**, not the front door.

Once you have a serious economic explanation, use `mechanism.py` to commit to:

- who is on the other side;
- why they must or strongly prefer to trade;
- an observable proxy for that flow;
- why arbitrage has not removed the edge;
- the bad state / falsifier that could prove the story wrong.

If the story was developed after looking at results, do **not** retroactively
upgrade the old hypothesis. Register a fresh OOS test in the same family:

```python
s2 = s.with_mechanism(
    mechanism,
    is_end="2025-12-31",
    oos_start="2026-01-01",
)
```

The family trial history stays intact. The old searches are never forgotten.

## What `Study.diagnose()` runs

The convenience bundle uses the existing rigorous diagnostics underneath:

- one-period-forward signal response / binscatter;
- Newey-West signal regression;
- drift decomposition (is this just a static long position?);
- IC decay across several horizons;
- leak-safe volatility targeting;
- breakeven transaction cost.

You can still call every low-level diagnostic directly when you need custom
research logic.

## Layout

| module | job |
|---|---|
| `study.py` | simple workflow: register → diagnose → test |
| `priors.py` | tier taxonomy (`RISK_PREMIUM` → `UNLABELED`) and trial multipliers |
| `mechanism.py` | typed counterparty + persistence; falsifiers that **cap** the asserted tier |
| `registry.py` | SQLite pre-registration, append-only trial log |
| `evaluate.py` | stationary bootstrap CIs, deflated Sharpe fed by the real trial count |
| `diagnostics.py` | leak-safe vol targeting, binscatter response, decay, breakeven cost, null band, drift decomposition |
| `cli.py` | guided `edgelab new` and `edgelab show` commands |

## Advanced workflow

The original lower-level API remains supported:

1. Optionally write and test a `Mechanism` before the decisive OOS test.
2. `registry.register(...)` the exact claim and split.
3. Run diagnostics on the signal.
4. `evaluate(...)` net-of-cost PnL.
5. After a sweep completes, `redeflate(...)` the surviving candidate against
   the family's **final** trial count.

## Known gaps

- Feature versioning is still a string. Hash your upstream config into it, or
  refitting a model can silently invalidate prior trials.
- Mechanism revisions are not automatically counted as a separate search
  dimension beyond the new hypothesis/trials you register. Keep revisions in
  the same family and use a fresh OOS split.
- No PIT panel layer yet. Nothing guarantees a feature was available at the
  decision timestamp — a common killer in options/equity research.

## Demos

```text
python demo_study.py        # easiest way to learn EdgeLab
python demo.py              # a mined winner dies; a real edge is marginal
python demo_mechanism.py    # identical stories, only one survives its falsifier
python demo_diagnostics.py  # binscatter, decay, breakeven cost, null band
```
