"""Render the knob/theory analysis charts from data/analysis/*.csv into charts/analysis/."""
import csv, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("charts/analysis", exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "font.size": 11})

def load(name):
    path = f"data/analysis/{name}.csv"
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))

def f(r, k):
    try: return float(r[k])
    except: return np.nan

EXPS = ["explore", "rw", "temp", "loc", "target", "cullp"]
ACCENT = "#e76f51"; TEAL = "#2a9d8f"; PURP = "#5e548e"; GOLD = "#e9c46a"

# ---- 1. Pareto: coverage vs evenness, all single-knob configs + best combos ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
allrows = []
for e in EXPS:
    for r in load(e):
        allrows.append((e, r))
best = load("best")

for ax, (xk, xl, invert) in zip(axes, [("evenness", "angular evenness (1 = uniform)", False),
                                       ("gap", "dead-zone gap (0 = ideal)", True)]):
    for e in EXPS:
        rows = [r for (ee, r) in allrows if ee == e]
        xs = [f(r, xk) for r in rows]; ys = [f(r, "cov_mean") for r in rows]
        ax.plot(xs, ys, "-o", ms=5, lw=1, alpha=0.8, label=e)
    # best combos as stars
    bx = [f(r, xk) for r in best]; by = [f(r, "cov_mean") for r in best]
    ax.scatter(bx, by, marker="*", s=160, c="black", zorder=5, label="best combos")
    for r in best:
        ax.annotate(r["knob"], (f(r, xk), f(r, "cov_mean")), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(xl); ax.set_ylabel("multimodal coverage (cov_mean)")
    if invert: ax.invert_xaxis()
    ax.set_title(f"Coverage vs {xk}")
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("The central tradeoff: multi-basin coverage vs even spread (each point = one config)", y=1.01)
fig.tight_layout()
fig.savefig("charts/analysis/pareto.png", bbox_inches="tight")
plt.close(fig)
print("wrote charts/analysis/pareto.png")

# ---- 2. Per-knob small multiples: knob value vs cov_mean (left axis) + evenness (right) ----
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
titles = {"explore": "explore_floor", "rw": "relative_width", "temp": "temperature",
          "loc": "mate_locality", "target": "target (pool size)", "cullp": "cull_pressure"}
for ax, e in zip(axes.flat, EXPS):
    rows = load(e)
    if not rows:
        ax.set_visible(False); continue
    xs = [f(r, "value") for r in rows]
    cov = [f(r, "cov_mean") for r in rows]
    even = [f(r, "evenness") for r in rows]
    gap = [f(r, "gap") for r in rows]
    ax.plot(xs, cov, "-o", color=ACCENT, lw=1.8, label="coverage")
    ax.set_xlabel(titles[e]); ax.set_ylabel("coverage", color=ACCENT)
    ax.tick_params(axis="y", labelcolor=ACCENT)
    ax2 = ax.twinx()
    ax2.plot(xs, even, "-s", color=TEAL, lw=1.6, label="evenness")
    ax2.plot(xs, gap, "-^", color=PURP, lw=1.2, alpha=0.7, label="gap")
    ax2.set_ylabel("evenness / gap", color=TEAL); ax2.tick_params(axis="y", labelcolor=TEAL)
    ax2.grid(False); ax2.set_ylim(-0.02, 1.02)
    ax.set_title(titles[e])
fig.suptitle("Per-knob effect — coverage (orange, left) vs evenness (teal) & gap (purple, right)", y=1.0)
fig.tight_layout()
fig.savefig("charts/analysis/knobs.png", bbox_inches="tight")
plt.close(fig)
print("wrote charts/analysis/knobs.png")

# ---- 3. Method comparison ----
comp = load("compare")
if comp:
    methods = [r["method"] for r in comp]
    cov = [f(r, "cov_mean") for r in comp]
    even = [f(r, "evenness") for r in comp]
    gap = [f(r, "gap") for r in comp]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.25
    ax.bar(x - w, [c / max(cov) for c in cov], w, label="coverage (norm)", color=ACCENT)
    ax.bar(x, even, w, label="evenness", color=TEAL)
    ax.bar(x + w, gap, w, label="gap", color=PURP)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_title("Resource vs SSEA vs Speciated — coverage(norm), evenness, gap")
    for i, c in enumerate(cov):
        ax.annotate(f"{c:.0f}", (i - w, c / max(cov)), ha="center", va="bottom", fontsize=8)
    ax.legend(); fig.tight_layout()
    fig.savefig("charts/analysis/methods.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/analysis/methods.png")

# ---- 4. Best combos ranked by composite score ----
if best:
    def score(r): return f(r, "cov_mean") / 22.0 + f(r, "evenness") - 3 * f(r, "gap")
    ranked = sorted(best, key=score, reverse=True)
    names = [r["knob"] for r in ranked]; sc = [score(r) for r in ranked]
    cols = [ACCENT if "explore" in n and "cullp" in n else TEAL if n in ("cullp4","baseline") else GOLD for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names)), sc, color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("composite score = cov/22 + evenness − 3·gap")
    ax.set_title("Best combos ranked (balance of coverage + even spread)")
    for i, (r, s) in enumerate(zip(ranked, sc)):
        ax.annotate(f"cov {f(r,'cov_mean'):.0f}  even {f(r,'evenness'):.2f}  gap {f(r,'gap'):.3f}",
                    (s, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig("charts/analysis/best.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/analysis/best.png")
