"""
P-value check (Oct 5): rerun the phase permutation tests with the shuffled null
averaged over the SAME number of CV splits (3) as the real score.

Before, the real score averaged 3 splits but each shuffle used 1. That makes the
null noisier than the real score. Usually that makes the test conservative, but
it is not like-for-like, so this redoes it properly.

Usage: python pfix_phase.py <job> <n_perm>
  jobs: cue_zp_fp, cue_causal_fp, cue_causal_3c, glove_fp
"""
import json, sys
import numpy as np
import common as C

job, N = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(11)
R = 3  # repeats for BOTH real and null

if job.startswith("cue"):
    src = open("phase_permutation.py").read().split("MAKE, RELEASE, MAKE_EQ")[0]
    g = {"__name__": "pp"}
    sys_argv = sys.argv
    sys.argv = ["x", "1", "1"]
    exec(compile(src, "pp_head", "exec"), g)
    sys.argv = sys_argv
    tg_features, sel, blocks, deficit, labels, on = (g[k] for k in ["tg_features", "sel", "blocks", "deficit", "labels", "on"])
    causal = "causal" in job
    X = tg_features(causal, on)
    keep = labels != 3 if job.endswith("fp") else labels > 0
    X, yy = X[keep], labels[keep]
    A, B = sel((0.5, 1.5)), sel((2.3, 3.0))
    Ae = sel((0.5, 1.2))
    lab_fun = lambda y: blocks(X, y, A, B, repeats=R)
    real = blocks(X, yy, A, B, repeats=R)

    def swap_def():
        Xs = X.copy()
        sw = rng.random(len(yy)) < .5
        ch = np.arange(X.shape[1])
        Xs[np.ix_(sw, ch, Ae)] = X[np.ix_(sw, ch, B)]
        Xs[np.ix_(sw, ch, B)] = X[np.ix_(sw, ch, Ae)]
        return deficit(blocks(Xs, yy, Ae, B, repeats=R))
    real_def = deficit(blocks(X, yy, Ae, B, repeats=R))
else:
    src = open("glove_aligned_phase.py").read().split("summary = {}")[0]
    g = {"__name__": "gp"}
    sys_argv = sys.argv
    sys.argv = ["x", "1"]
    exec(compile(src, "gp_head", "exec"), g)
    sys.argv = sys_argv
    blocks, deficit, Xm, Xr, yy = (g[k] for k in ["blocks", "deficit", "Xm", "Xr", "yy"])
    lab_fun = lambda y: blocks(Xm, Xr, y, repeats=R)
    real = blocks(Xm, Xr, yy, repeats=R)

    def swap_def():
        sw = rng.random(len(yy)) < .5
        A2, B2 = Xm.copy(), Xr.copy()
        A2[sw], B2[sw] = Xr[sw], Xm[sw]
        return deficit(blocks(A2, B2, yy, repeats=R))
    real_def = deficit(real)

null = [lab_fun(rng.permutation(yy)) for _ in range(N)]
dnull = np.array([swap_def() for _ in range(N)])
out = {"job": job, "n_perm": N, "repeats_real_and_null": R, "blocks": {}}
print(f"[{job}] {N} permutations, {R} CV repeats for real AND null")
for k in real:
    nv = np.array([n[k] for n in null])
    p = float((np.sum(nv >= real[k]) + 1) / (N + 1))
    out["blocks"][k] = dict(acc=float(real[k]), null95=float(np.percentile(nv, 95)), p=p)
    print(f"  {k:16s} {real[k]*100:5.1f}%  shuffled 95th {np.percentile(nv,95)*100:4.1f}%  p={p:.3f}")
pd = float((np.sum(dnull >= real_def) + 1) / (N + 1))
out["deficit"] = dict(value=float(real_def), null95=float(np.percentile(dnull, 95)), p=pd)
print(f"  transfer gap {real_def*100:+.1f} pts  null 95th {np.percentile(dnull,95)*100:+.1f}  p={pd:.3f}")
json.dump(out, open(f"results/pfix_{job}.json", "w"), indent=2)
