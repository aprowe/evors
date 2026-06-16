"""Render evors benchmark charts from the CSVs in ./data into ./charts."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

CLASSIC = "#d1495b"   # red
SSEA = "#2a9d8f"      # teal


def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(x) if x not in ("", "NaN", "nan") else np.nan
                      for x in r] for r in rows[1:]])
    return header, data


# ---------------- 1. XOR convergence ----------------
hdr, d = load_csv("data/xor_convergence.csv")
trial = d[:, 0]
classic = d[:, 1]
ssea = d[:, 2]

def first_hit(arr, thr=0.98):
    idx = np.where(arr >= thr)[0]
    return int(idx[0]) if len(idx) else None

c_hit, s_hit = first_hit(classic), first_hit(ssea)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(trial, classic, color=CLASSIC, lw=1.8, label="ClassicPopulation")
ax.plot(trial, ssea, color=SSEA, lw=1.8, label="SoftStochastic (SSEA)")
ax.axhline(0.98, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(trial[-1], 0.982, "0.98 solve threshold", ha="right", va="bottom",
        color="gray", fontsize=9)
for hit, col, name, ytext in [(c_hit, CLASSIC, "classic", 0.70),
                              (s_hit, SSEA, "ssea", 0.58)]:
    if hit is not None:
        ax.axvline(trial[hit], color=col, ls=":", lw=1.2, alpha=0.8)
        ax.annotate(f"{name} solved @ {int(trial[hit]):,} evals",
                    xy=(trial[hit], 0.98), xytext=(trial[-1] * 0.30, ytext),
                    color=col, fontsize=9, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=col, alpha=0.7),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.9))
ax.set_xlabel("evaluations (trials)")
ax.set_ylabel("best fitness  (1 − MSE)")
ax.set_title("XOR convergence — best fitness vs evaluations")
ax.set_ylim(0.5, 1.01)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("charts/xor_convergence.png")
print("wrote charts/xor_convergence.png  classic_hit=%s ssea_hit=%s" % (c_hit, s_hit))

# Early-window zoom (first 12k evals) to show the race more clearly.
fig, ax = plt.subplots(figsize=(9, 4.6))
w = trial <= 12000
ax.plot(trial[w], classic[w], color=CLASSIC, lw=1.8, label="ClassicPopulation")
ax.plot(trial[w], ssea[w], color=SSEA, lw=1.8, label="SoftStochastic (SSEA)")
ax.axhline(0.98, color="gray", ls="--", lw=1, alpha=0.7)
ax.set_xlabel("evaluations (trials)")
ax.set_ylabel("best fitness  (1 − MSE)")
ax.set_title("XOR convergence — first 12,000 evaluations (zoom)")
ax.set_ylim(0.5, 1.01)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("charts/xor_convergence_zoom.png")
print("wrote charts/xor_convergence_zoom.png")


# ---------------- 2. Circle solution spread ----------------
_, dc = load_csv("data/circle_classic.csv")
_, ds = load_csv("data/circle_ssea.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), subplot_kw={"aspect": "equal"})
theta = np.linspace(0, 2 * np.pi, 400)
for ax, data, title, col in [
    (axes[0], dc, "ClassicPopulation", CLASSIC),
    (axes[1], ds, "SoftStochastic (SSEA)", SSEA),
]:
    ax.plot(np.cos(theta), np.sin(theta), color="gray", lw=1, ls="--", alpha=0.7)
    sc = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="viridis",
                    s=14, vmin=0, vmax=1, alpha=0.8, edgecolors="none")
    n_hi = int(np.sum(data[:, 2] >= 0.95))
    ax.set_title(f"{title}\n{n_hi}/{len(data)} samples on the ring (fit≥0.95)")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.colorbar(sc, ax=axes, shrink=0.8, label="fitness  1−(x²+y²−1)²")
fig.suptitle("Unit-circle benchmark — solution spread (whole ring is optimal)", y=1.02)
fig.savefig("charts/circle_spread.png", bbox_inches="tight")
print("wrote charts/circle_spread.png")


# ---------------- 3. Angular coverage histogram ----------------
def angles_of_high(data, thr=0.95):
    hi = data[data[:, 2] >= thr]
    if len(hi) == 0:
        return np.array([])
    a = np.arctan2(hi[:, 1], hi[:, 0])
    a[a < 0] += 2 * np.pi
    return a

bins = 12
edges = np.linspace(0, 2 * np.pi, bins + 1)
ca, sa = angles_of_high(dc), angles_of_high(ds)
ch, _ = np.histogram(ca, bins=edges)
sh, _ = np.histogram(sa, bins=edges)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2),
                         subplot_kw={"projection": "polar"})
width = 2 * np.pi / bins
centers = edges[:-1] + width / 2
for ax, h, title, col in [
    (axes[0], ch, f"ClassicPopulation\n{int(np.sum(ch>0))}/12 sectors covered", CLASSIC),
    (axes[1], sh, f"SoftStochastic (SSEA)\n{int(np.sum(sh>0))}/12 sectors covered", SSEA),
]:
    ax.bar(centers, h, width=width * 0.9, color=col, alpha=0.85, edgecolor="white")
    ax.set_title(title, pad=18)
    ax.set_theta_zero_location("E")
    ax.set_yticklabels([])
fig.suptitle("Angular coverage of on-ring solutions (12 sectors)", y=1.04)
fig.tight_layout()
fig.savefig("charts/circle_angular.png", bbox_inches="tight")
print("wrote charts/circle_angular.png")


# ---------------- 4. NEAT XOR: fitness + topology growth ----------------
if os.path.exists("data/neat_xor.csv"):
    _, dn = load_csv("data/neat_xor.csv")
    tr, bestf, hidden = dn[:, 0], dn[:, 1], dn[:, 2]
    NEAT = "#5e548e"
    fig, ax1 = plt.subplots(figsize=(9, 5.0))
    ax1.plot(tr, bestf, color=NEAT, lw=1.8, label="best fitness")
    ax1.axhline(0.98, color="gray", ls="--", lw=1, alpha=0.7)
    ax1.set_xlabel("evaluations (trials)")
    ax1.set_ylabel("best fitness  (1 − MSE)", color=NEAT)
    ax1.tick_params(axis="y", labelcolor=NEAT)
    ax1.set_ylim(0.5, 1.01)
    ax2 = ax1.twinx()
    ax2.step(tr, hidden, color="#e07a5f", lw=1.6, where="post", label="champion hidden nodes")
    ax2.set_ylabel("champion hidden-node count", color="#e07a5f")
    ax2.tick_params(axis="y", labelcolor="#e07a5f")
    ax2.set_ylim(-0.2, max(3, hidden.max() + 1))
    ax2.grid(False)
    hit = first_hit(bestf)
    title = "NEAT on SSEA — XOR: fitness vs evolved topology"
    if hit is not None:
        title += f"  (solved @ {int(tr[hit]):,} evals)"
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig("charts/neat_xor.png")
    print("wrote charts/neat_xor.png")

# ---------------- bench: convergence + coverage + scatter ----------------
import glob
if os.path.isdir("data/bench"):
    os.makedirs("charts/bench", exist_ok=True)

    # Per-problem convergence (one line per config).
    for path in glob.glob("data/bench/*_convergence.csv"):
        name = os.path.basename(path).replace("_convergence.csv", "")
        with open(path) as f:
            rows = list(csv.reader(f))
        header, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
        if data.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        evals = data[:, 0]
        for j, label in enumerate(header[1:], start=1):
            ax.plot(evals, data[:, j], lw=1.4, label=label)
        ax.set_xlabel("evaluations")
        ax.set_ylabel("best-so-far objective (min)")
        ax.set_title(f"{name} — convergence")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"charts/bench/{name}_convergence.png")
        plt.close(fig)
    print("wrote charts/bench/*_convergence.png")

    # Coverage bars from results.csv (classic vs ssea max per problem).
    if os.path.exists("data/bench/results.csv"):
        with open("data/bench/results.csv") as f:
            rows = list(csv.DictReader(f))
        probs = sorted({r["problem"] for r in rows})
        def best_cov(prob, kind):
            vals = [float(r["mean_coverage"]) for r in rows
                    if r["problem"] == prob and r["config"].startswith(kind)]
            return max(vals) if vals else 0.0
        classic = [best_cov(p, "classic") for p in probs]
        ssea = [best_cov(p, "ssea") for p in probs]
        x = np.arange(len(probs))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - 0.2, classic, 0.4, label="ClassicPopulation", color="#d1495b")
        ax.bar(x + 0.2, ssea, 0.4, label="SoftStochastic", color="#2a9d8f")
        ax.set_xticks(x); ax.set_xticklabels(probs, rotation=30, ha="right")
        ax.set_ylabel("basins covered (mean)")
        ax.set_title("Basin coverage — Classic vs SSEA")
        ax.legend()
        fig.tight_layout()
        fig.savefig("charts/bench/coverage.png")
        plt.close(fig)
        print("wrote charts/bench/coverage.png")

    # Final-pool scatter (best classic vs best ssea config) for multimodal probs.
    for path in glob.glob("data/bench/*_scatter.csv"):
        name = os.path.basename(path).replace("_scatter.csv", "")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        # pick one classic and one ssea config present in the scatter.
        def pts(kind):
            xs = [(float(r["x"]), float(r["y"])) for r in rows if r["config"].startswith(kind)]
            return np.array(xs) if xs else np.empty((0, 2))
        c, s = pts("classic"), pts("ssea")
        if len(c) == 0 and len(s) == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), subplot_kw={"aspect": "equal"})
        for ax, data, title, col in [
            (axes[0], c, "ClassicPopulation", "#d1495b"),
            (axes[1], s, "SoftStochastic", "#2a9d8f"),
        ]:
            if len(data):
                ax.scatter(data[:, 0], data[:, 1], s=10, c=col, alpha=0.6, edgecolors="none")
            ax.set_title(title)
        fig.suptitle(f"{name} — final population spread")
        fig.tight_layout()
        fig.savefig(f"charts/bench/{name}_scatter.png")
        plt.close(fig)
    print("wrote charts/bench/*_scatter.png")
