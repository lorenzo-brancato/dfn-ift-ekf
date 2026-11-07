"""Summary figures of the 21-cell Mohtat EKF batch (scripts/mohtat_batch.py summary.csv), one or two configurations.

    python scripts/mohtat_batch_figs.py results/mohtat/batch_RfLLI/summary.csv:"Rf, LLI" \
        results/mohtat/batch_RfLAMLLI/summary.csv:"Rf, LAM_n, LLI" --out results/mohtat/batch_figs

fig_parity   SOH, LLI [A.h] and LAM_n estimated against the references (C/20 SOH; electrode-window fit of the nearest
             check-up), every cell and state of life; colour = chamber (cold / room / hot), marker = state of life.
fig_errors   per cell and state of life: SOH error [points], SOC RMS error [%], innovation RMS [mV].
fig_rf       the film-resistance multiplier against the reference SOH, per cell.
Cells are Room | Cold | Hot within each triple (01-03, 04-06, ...): 01-03 C/5, 04-06 1.5C, 07-09 2C, 10-12 C/5 charge 1.5C
discharge, 13-15 half DoD C/5, 16-18 half DoD C/5 + 1.5C, 19-21 half DoD 1.5C + drive cycle.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4
CHAMBER = {0: ("room", "#009E73"), 1: ("cold", "#0072B2"), 2: ("hot", "#D55E00")}
STATE = {"BOL": "o", "SOH90": "s", "SOH80": "^"}
PROTO = ["C/5", "1.5C", "2C", "C/5-1.5C", "½DoD C/5", "½DoD C/5-1.5C", "½DoD drive"]


def chamber(cell):
    return CHAMBER[(int(cell[1:]) - 1) % 3]


def load(specs):
    out = []
    for s in specs:
        path, _, label = s.partition(":")
        d = pd.read_csv(path)
        d["config"] = label or os.path.basename(os.path.dirname(path))
        out.append(d)
    return out


def parity(ax, d, x, y, fill=True, label_states=False):
    for _, r in d.iterrows():
        if not r.get("finite", True) or not np.isfinite(r.get(x, np.nan)) or not np.isfinite(r.get(y, np.nan)):
            continue
        _, col = chamber(r.cell)
        ax.plot(r[x], r[y], STATE[r.state], color=col, mfc=col if fill else "none", ms=4.5, mew=0.9, alpha=0.9)


def legend_handles(configs):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="none", marker="o", color=c, label=n) for n, c in CHAMBER.values()]
    h += [Line2D([], [], ls="none", marker=m, color="0.3", mfc="none", label=s) for s, m in STATE.items()]
    if len(configs) > 1:
        h += [Line2D([], [], ls="none", marker="o", color="0.3", label=configs[0]),
              Line2D([], [], ls="none", marker="o", color="0.3", mfc="none", label=configs[1])]
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="summary.csv[:label]; the first is drawn filled, the second open")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mohtat", "batch_figs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load(a.runs)
    labels = [d.config.iloc[0] for d in D]

    # ---- parity: SOH, LLI, LAM_n
    fig, ax = plt.subplots(1, 3, figsize=(170 * mm, 62 * mm), layout="constrained")
    for i, d in enumerate(D):
        parity(ax[0], d, "SOH_ref", "SOH", fill=(i == 0))
        parity(ax[1], d, "LLI_ref_Ah", "LLI_Ah", fill=(i == 0))
        if "LAM_n" in d and np.nanmax(np.abs(d.LAM_n.values)) > 0:
            parity(ax[2], d.assign(LAM_n=100 * d.LAM_n, LAM_n_ref=100 * d.LAM_n_ref), "LAM_n_ref", "LAM_n", fill=(i == 0))
    allv = pd.concat(D)
    ext = lambda cols, lo, hi: (min(lo, np.nanmin(allv[cols].values)), max(hi, np.nanmax(allv[cols].values)))
    s_lo, s_hi = ext(["SOH", "SOH_ref"], 60, 102)
    l_lo, l_hi = ext(["LLI_Ah", "LLI_ref_Ah"], -0.1, 2.0)
    for k, (lo, hi, lab) in enumerate(((s_lo - 2, s_hi + 2, "SOH [%]"), (l_lo - 0.1, l_hi + 0.1, "LLI [A h]"), (-35, 40, r"LAM$_n$ [%]"))):
        ax[k].plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
        ax[k].set_xlim(lo, hi); ax[k].set_ylim(lo, hi)
        ax[k].set_xlabel("reference " + lab); ax[k].set_ylabel("EKF " + lab)
        ax[k].spines[["top", "right"]].set_visible(False)
    ax[0].set_title("(a) SOH", fontsize=9); ax[1].set_title("(b) loss of lithium inventory", fontsize=9)
    ax[2].set_title(r"(c) LAM$_n$ (only where estimated)", fontsize=9)
    fig.legend(handles=legend_handles(labels), loc="outside lower center", ncol=8, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "fig_parity." + ext))
    plt.close(fig)

    # ---- per-cell errors
    cells = sorted(set(D[0].cell))
    xs = np.arange(len(cells))
    fig, ax = plt.subplots(3, 1, figsize=(170 * mm, 140 * mm), layout="constrained", sharex=True)
    width = 0.8 / (3 * len(D))
    for i, d in enumerate(D):
        for j, st in enumerate(STATE):
            dd = d[d.state == st].set_index("cell").reindex(cells)
            off = (i * 3 + j - (3 * len(D) - 1) / 2) * width
            hatch = None if i == 0 else "////"
            col = ["#9E9E9E", "#56B4E9", "#E69F00"][j]
            ax[0].bar(xs + off, dd.SOH - dd.SOH_ref, width, color=col, hatch=hatch, edgecolor="k", lw=0.3,
                      label="%s, %s" % (st, d.config.iloc[0]))
            ax[1].bar(xs + off, dd.SOC_rms, width, color=col, hatch=hatch, edgecolor="k", lw=0.3)
            ax[2].bar(xs + off, dd.innov_mV, width, color=col, hatch=hatch, edgecolor="k", lw=0.3)
            bad = dd.index[~dd.finite.fillna(False).astype(bool)]
            for b in bad:
                ax[0].text(cells.index(b) + off, 0, "x", color="r", ha="center", va="center", fontsize=8)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("SOH error [points]"); ax[1].set_ylabel("SOC RMS error [%]"); ax[2].set_ylabel("innovation RMS [mV]")
    ax[1].set_yscale("log"); ax[2].set_yscale("log")
    for k in range(3):
        ax[k].spines[["top", "right"]].set_visible(False)
        for g in range(1, 7):
            ax[k].axvline(3 * g - 0.5, color="0.8", lw=0.5)
    ax[2].set_xticks(xs, ["%s\n%s" % (c, chamber(c)[0]) for c in cells], fontsize=7)
    for g, p in enumerate(PROTO):
        ax[0].text(3 * g + 1, ax[0].get_ylim()[1], p, ha="center", va="top", fontsize=7)
    ax[0].legend(ncol=3 * len(D) // 2 if len(D) > 1 else 3, fontsize=7, frameon=False, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "fig_errors." + ext))
    plt.close(fig)

    # ---- Rf against SOH
    fig, ax = plt.subplots(1, len(D), figsize=(170 * mm, 65 * mm), layout="constrained", squeeze=False, sharey=True)
    for i, d in enumerate(D):
        for c in cells:
            dd = d[(d.cell == c) & d.finite.fillna(False).astype(bool)].sort_values("SOH_ref", ascending=False)
            if len(dd):
                ax[0, i].plot(dd.SOH_ref, dd.Rf, "-o", color=chamber(c)[1], lw=0.8, ms=3, alpha=0.85)
        ax[0, i].set_yscale("log"); ax[0, i].invert_xaxis()
        ax[0, i].set_xlabel("reference SOH [%]"); ax[0, i].set_title(d.config.iloc[0], fontsize=9)
        ax[0, i].spines[["top", "right"]].set_visible(False)
    ax[0, 0].set_ylabel(r"$R_f / R_{f,\mathrm{BOL}}$ (median)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "fig_rf." + ext))
    plt.close(fig)

    # ---- table of aggregate errors
    rows = []
    for d in D:
        ok = d[d.finite.fillna(False).astype(bool)]
        for st in list(STATE) + ["all"]:
            s = ok if st == "all" else ok[ok.state == st]
            rows.append(dict(config=d.config.iloc[0], state=st, n_ok=len(s), n_fail=int((~d.finite.fillna(False).astype(bool)).sum()) if st == "all" else
                             int((~d[d.state == st].finite.fillna(False).astype(bool)).sum()),
                             SOH_MAE=np.mean(np.abs(s.SOH - s.SOH_ref)), SOH_max=np.max(np.abs(s.SOH - s.SOH_ref)) if len(s) else np.nan,
                             LLI_MAE_Ah=np.mean(np.abs(s.LLI_Ah - s.LLI_ref_Ah)), SOC_rms_med=np.median(s.SOC_rms),
                             innov_med_mV=np.median(s.innov_mV)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "aggregate.csv"), index=False)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
