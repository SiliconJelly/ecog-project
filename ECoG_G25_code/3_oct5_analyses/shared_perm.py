"""Label-permutation null for the shared-channel cross-phase transfer in shared_onset.py
(same procedure, channel selection inside training folds, k = 5 and 10)."""
import sys, json
import numpy as np
src = open("shared_onset.py").read()
head = src.split("# per-channel transfer")[0]
fun = "def subset_transfer" + src.split("def subset_transfer")[1].split("sub = {k:")[0]
g = {"__name__": "so"}; sys.argv = ["x", "1"]
exec(compile(head, "h", "exec"), g)
exec(compile(fun, "f", "exec"), g)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N = 200
rng = np.random.default_rng(9)
yy0 = g["yy"].copy()
out = {}
for k in (5, 10):
    real = g["subset_transfer"](k)
    null = []
    for _ in range(N):
        g["yy"] = rng.permutation(yy0)
        null.append(g["subset_transfer"](k, repeats=10))
    g["yy"] = yy0
    null = np.array(null)
    p = float((np.sum(null >= real) + 1) / (N + 1))
    out[k] = dict(real=real, null_mean=float(null.mean()), null95=float(np.percentile(null, 95)), p=p)
    print(f"k={k}: transfer {real*100:.1f}%  shuffled mean {null.mean()*100:.1f}%, 95th {np.percentile(null,95)*100:.1f}%  p={p:.3f}", flush=True)
json.dump(out, open("results/shared_perm.json", "w"), indent=2)
