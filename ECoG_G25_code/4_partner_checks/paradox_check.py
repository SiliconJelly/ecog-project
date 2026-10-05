"""
Check of the partner's pose_state_paradox.py (Oct 5). Same features and windows as his
script, then three fixes:
 1. Permutation test (label shuffles, 200) - his script reports CV accuracy with no null.
 2. Rest windows overlap (0.5 s windows every 0.25 s) and several come from the same
    rest gap; CV is redone with all windows from one gap kept in the same fold.
 3. Posture check: how different are the 'fist-like rest' windows from a real fist?
    And a glove-only decoder on the same windows (if the glove separates them, pose is not matched).
"""
import numpy as np, json, sys, os
os.chdir("/home/claude/partner/ecog_team_whitening")
src = open("pose_state_paradox.py").read().split("fig = plt.figure")[0]
import matplotlib; matplotlib.use("Agg")
g = {"__name__": "pp"}
exec(compile(src, "pp", "exec"), g)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
rng = np.random.default_rng(0)
offsets, onsets, rest_centers = g["offsets"], g["onsets"], g["rest_centers"]
gap_id = np.searchsorted(onsets, rest_centers)          # which rest gap each window came from
def acc(X, y, groups=None, seeds=range(5)):
    m = lambda: make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    if groups is None:
        return np.mean([cross_val_score(m(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean() for s in seeds])
    return np.mean([cross_val_score(m(), X, y, groups=groups, cv=StratifiedGroupKFold(5, shuffle=True, random_state=s)).mean() for s in seeds])
out = {}
for name, gest_key, idx, tmpl in [("open", 3, g["open_rest_idx"], g["open_template"]), ("fist", 1, g["fist_rest_idx"], g["fist_template"])]:
    Xg = np.array([h for _, h, _ in g["gesture"][gest_key]]); Xr = g["rest_hg"][idx]
    Gg = np.array([gl for gl, _, _ in g["gesture"][gest_key]]); Gr = g["rest_glove"][idx]
    X = np.r_[Xg, Xr]; y = np.r_[np.ones(len(Xg)), np.zeros(len(Xr))]
    G = np.r_[Gg, Gr]
    # groups: each cued trial is its own group; rest windows grouped by gap
    grp = np.r_[np.arange(len(Xg)) + 10000, gap_id[idx]]
    real = acc(X, y); real_g = acc(X, y, grp)
    null = np.array([acc(X, rng.permutation(y), grp, seeds=[0]) for _ in range(200)])
    p = (np.sum(null >= real_g) + 1) / 201
    glove_acc = acc(G, y, grp)
    dist = np.linalg.norm(Gr.mean(0) - Gg.mean(0))
    n_gaps = len(np.unique(gap_id[idx]))
    out[name] = dict(his_style=real, grouped=real_g, null95=float(np.percentile(null, 95)), p=float(p), glove=glove_acc,
                     posture_gap=float(dist), rest_windows=int(len(idx)), distinct_gaps=int(n_gaps),
                     rest_posture=Gr.mean(0).tolist(), cued_posture=Gg.mean(0).tolist())
    print(f"[{name}] his-style CV {real*100:.1f}% | grouped CV {real_g*100:.1f}% (shuffle 95th {np.percentile(null,95)*100:.1f}%, p={p:.3f}) "
          f"| glove-only {glove_acc*100:.1f}% | {len(idx)} rest windows from {n_gaps} gaps | posture distance {dist:.2f}")
    print("    cued posture " + " ".join(f"{v:.2f}" for v in Gg.mean(0)) + " | rest posture " + " ".join(f"{v:.2f}" for v in Gr.mean(0)))
json.dump(out, open("/home/claude/work/results/paradox_check.json", "w"), indent=2)
