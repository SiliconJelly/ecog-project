"""
Add-on to glove_aligned_phase.py: is each glove-aligned block (including the
cross-phase ones, ~55%) above chance? Null = shuffled gesture labels.
Also redraws figures/T4_glove_aligned_phase.png with this result.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import importlib.util
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
# reuse the definitions without rerunning the slow part
src = open("glove_aligned_phase.py").read().split("summary = {}")[0]
g = {}
exec(compile(src, "glove_aligned_phase_head", "exec"), g)
blocks, Xm, Xr, yy, move_on, rel_on, ok, labels, rel_t = (g[k] for k in ["blocks", "Xm", "Xr", "yy", "move_on", "rel_on", "ok", "labels", "rel_t"])
rng = np.random.default_rng(7)
prev = json.load(open("results/glove_aligned_phase.json"))
real = prev["summary"]["glove-aligned"]["blocks"]
null = [blocks(Xm, Xr, rng.permutation(yy), repeats=1) for _ in range(N)]
res = {}
for k in real:
    nv = np.array([n[k] for n in null])
    res[k] = dict(acc=real[k], null95=float(np.percentile(nv, 95)), p=float((np.sum(nv >= real[k]) + 1) / (N + 1)))
    print(f"  {k:16s} {real[k]*100:5.1f}%  shuffled 95th {np.percentile(nv,95)*100:.1f}%  p={res[k]['p']:.3f}")
prev["label_permutation_glove_aligned"] = res
json.dump(prev, open("results/glove_aligned_phase.json", "w"), indent=2, default=float)

M = np.load("results/glove_aligned_tg.npy"); nb = M.shape[0]
summ = prev["summary"]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25, 1]})
for k, col, n in [(1, "#2a78d6", "Fist"), (2, "#eb6834", "Peace")]:
    m = labels[ok] == k
    ax[0].scatter(move_on[ok][m], rel_on[ok][m] - 2.0, color=col, s=25, alpha=.8, label=n)
ax[0].set(xlabel="Movement onset (s after cue on)", ylabel="Release onset (s after cue OFF)",
          title="A. When the hand actually moved (glove)")
ax[0].legend(frameon=False)
im = ax[1].imshow(M, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=.5, vmin=0.2, vmax=1), extent=[0, nb, 0, nb])
ax[1].axvline(nb / 2, color="k", lw=1.2); ax[1].axhline(nb / 2, color="k", lw=1.2)
ticks = [int(np.argmin(abs(rel_t - t))) for t in (0, 0.5, 1.0)]
tk = np.array(ticks + [t + len(rel_t) for t in ticks]) + .5
ax[1].set_xticks(tk, ["0", ".5", "1"] * 2); ax[1].set_yticks(tk, ["0", ".5", "1"] * 2)
ax[1].text(nb * .25, nb * 1.01, "closing", ha="center", va="bottom"); ax[1].text(nb * .75, nb * 1.01, "opening", ha="center", va="bottom")
ax[1].set(xlabel="Test time: s from MOVEMENT onset | s from RELEASE onset", ylabel="Train time: closing | opening")
ax[1].set_title("B. Fist vs peace, glove-aligned generalization\n(red = above chance)", pad=18)
fig.colorbar(im, ax=ax[1], shrink=.8, label="Accuracy")
order = ["make_make", "release_release", "make_release", "release_make"]
for j, (name, col) in enumerate(zip(summ, ["#2a78d6", "#8a8a85"])):
    xs = np.arange(4) + (j - .5) * .38
    ax[2].bar(xs, [summ[name]["blocks"][k] * 100 for k in order], .35, color=col, label=name.split(" (")[0])
for i, k in enumerate(order):
    x = i - .19
    ax[2].plot([x - .17, x + .17], [res[k]["null95"] * 100] * 2, "k--", lw=1)
ax[2].axhline(50, color="k", lw=.6)
ax[2].set_xticks(range(4), ["close→close", "open→open", "close→open", "open→close"], fontsize=9)
ax[2].set(ylim=(40, 75), ylabel="Accuracy (%)",
          title=f"C. Within- vs cross-phase (dashed = shuffle 95th)\ntransfer gap: glove-aligned p={summ['glove-aligned']['p']:.3f}, cue-aligned p={summ['cue-aligned (shifted by median latency)']['p']:.3f}")
ax[2].legend(frameon=False)
for a in (ax[0], ax[2]):
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/T4_glove_aligned_phase.png", dpi=160)
