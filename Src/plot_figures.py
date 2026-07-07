#!/usr/bin/env python3
"""Regenerate the paper's trajectory figure from the per-iteration CSV logs.
Usage:  python3 plot_figures.py
Requires: matplotlib. Reads results/*.csv, writes figures/phase4_comparison.png.
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def load(name):
    rows = list(csv.DictReader(open(os.path.join(RES, name))))
    r = [int(x["r"]) for x in rows]
    ratio = [float(x["ratio"]) for x in rows]
    phi = [float(x["Phi"]) for x in rows] if rows and "Phi" in rows[0] else None
    return r, ratio, phi

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    rf, qf, ff = load("p4_fixed_hiddenPM_n2000_eps5.csv")
    ra, qa, fa = load("p4_adapt_hiddenPM_n2000_eps5.csv")
    ax1.plot(rf, qf, "o-", color="tab:gray", label="fixed-2 (paper)")
    ax1.plot(ra, qa, "s-", color="tab:red", label="adaptive (ours)")
    ax1.axhline(0.95, ls=":", c="k", lw=1, label="target 1-eps")
    ax1.set_xlabel("iteration"); ax1.set_ylabel("|M|/mu")
    ax1.set_title("hiddenPM n=2000, eps=0.05 - quality"); ax1.legend(); ax1.grid(alpha=.3)
    lm = math.log2(639042)
    if ff and fa:
        ax2.plot(range(1, len(ff)+1), ff, "o-", color="tab:gray", label="fixed-2")
        ax2.plot(range(1, len(fa)+1), fa, "s-", color="tab:red", label="adaptive")
    ax2.axhline(lm, ls=":", c="k", lw=1, label="log2 m (certificate fires)")
    ax2.set_xlabel("iteration"); ax2.set_ylabel("Phi (certificate)")
    ax2.set_title("Theorem 1 certificate progress"); ax2.legend(); ax2.grid(alpha=.3)
    ax2.set_xlim(0, 12)
    os.makedirs(FIG, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "phase4_comparison.png"), dpi=150)
    print("wrote figures/phase4_comparison.png")

if __name__ == "__main__":
    main()
