"""Sensitivity check for same_hand.py: PAPER vs REST with no posture caliper, the original
caliper, and a strict caliper; plus a search for fist-like hands outside rock trials."""
import numpy as np, json, sys
src = open("same_hand.py").read().split("comparisons = [")[0]
g = {"__name__": "sh"}; exec(compile(src, "sh", "exec"), g)
on, labels, glove, fs = g["on"], g["labels"], g["glove"], g["fs"]
out = {}
idx = np.where(labels == 3)[0]
W = g["W"]
pa = np.array([g["posture"](on[i], W["PAPER"]) for i in idx]); pb = np.array([g["posture"](on[i], W["REST"]) for i in idx])
sa = np.array([g["speed"](on[i], W["PAPER"]) for i in idx]); sb = np.array([g["speed"](on[i], W["REST"]) for i in idx])
d = np.linalg.norm(pa - pb, axis=1)
for name, keep in [("no caliper", np.ones(len(idx), bool)), ("caliper 0.10 + speed", (d < .10) & (abs(sa - sb) < .03)),
                   ("strict 0.05 + speed 0.015", (d < .05) & (abs(sa - sb) < .015))]:
    k = idx[keep]
    NA = np.array([g["neural"](on[i], W["PAPER"]) for i in k]); NB = np.array([g["neural"](on[i], W["REST"]) for i in k])
    GA, GB = np.c_[pa[keep], sa[keep]], np.c_[pb[keep], sb[keep]]
    rg = g["test"](GA, GB, n_perm=100); rb = g["test"](NA, NB, n_perm=100)
    out[name] = dict(n=int(len(k)), glove=rg, brain=rb)
    print(f"PAPER vs REST, {name:26s} n={len(k):2d}  glove {rg[0]*100:5.1f}% (p={rg[2]:.3f})   brain {rb[0]*100:5.1f}% (p={rb[2]:.3f})", flush=True)
# fist-like hands outside rock trials: index+middle+ring all bent > 0.6
gl = glove
cue = g["cue"]
rest_mask = cue == 0
fistlike = (gl[1] > .6) & (gl[2] > .5) & (gl[3] > .6)
print(f"Fist-like posture: {np.mean(fistlike[rest_mask])*100:.2f}% of rest samples vs {np.mean(fistlike[cue==1])*100:.1f}% of rock-cue samples")
out["fistlike_rest_pct"] = float(np.mean(fistlike[rest_mask]) * 100); out["fistlike_rock_pct"] = float(np.mean(fistlike[cue == 1]) * 100)
json.dump(out, open("results/same_hand_sens.json", "w"), indent=2, default=float)
