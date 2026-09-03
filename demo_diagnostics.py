import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from edgelab.diagnostics import *

rng = np.random.default_rng(3)
N = 1500
# weak real signal: IC ~ 0.05, plus a pure-noise control
sig = rng.normal(0,1,N)
ret = 0.0004 + 0.010*rng.normal(0,1,N)
ret[1:] += 0.0009*sig[:-1]   # signal at t-1 predicts return at t
noise_sig = rng.normal(0,1,N)

for name, s in [("REAL (IC~0.06)", sig), ("NOISE", noise_sig)]:
    vt = vol_target(s, ret, target_vol=0.10, cost_per_turnover=0.0002)
    rc = signal_response(s[:-1], ret[1:], n_buckets=6)
    print(f"--- {name} ---")
    print(rc.summary())
    print("  decay:", {h: round(v,3) for h,v in decay_profile(s, ret).items()})
    print(f"  breakeven cost {breakeven_cost(vt):.1f} bps/turnover")
    print(f"  realised vol {vt.realised_vol:.3f}  net Sharpe "
          f"{vt.net_returns.mean()/vt.net_returns.std()*np.sqrt(252):+.2f}\n")

vt = vol_target(sig, ret, cost_per_turnover=0.0002)
eq = equity_curve(vt.net_returns)
p5,p50,p95 = null_band(vt.net_returns)
print("terminal: strategy %.3f | null p95 %.3f | inside band? %s"
      % (eq[-1], p95[-1], eq[-1] < p95[-1]))
np.save('/tmp/eq.npy', np.vstack([eq,p5,p50,p95]))
