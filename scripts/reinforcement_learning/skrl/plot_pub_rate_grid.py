"""
Publication-ready success-rate robustness grid (Cart + Shadow-Hand x 5 fault sweeps).

Same print-size / palette / legend conventions as plot_pub_reward_grid.py.
No std shading here: the source .txt tables report a single rate percentage
per config (fraction of episodes that succeeded), not a mean±std for it.

Include in LaTeX as:
    \\begin{figure*}[t]
      \\centering
      \\includegraphics[width=\\textwidth]{results/pub_rate_grid.pdf}
      \\caption{...}
    \\end{figure*}
"""
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ── print / style setup ──────────────────────────────────────────────────
FIG_W, FIG_H = 7.16, 2.6  # IEEE two-column \textwidth, 2 rows
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

STDS = [0.0, 1.0, 2.0, 5.0, 10.0]
MUS = [-1.0, 0.0, 1.0]
FAULT_XVALS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
DELAY_XVALS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]

ALGO_ORDER = ["MAPPO", "IPPO", "Adv-MAPPO"]
# Okabe-Ito colourblind-safe palette
COLORS = {"MAPPO": "#0072B2", "IPPO": "#009E73", "Adv-MAPPO": "#D55E00"}

TASKS = [("cart", "Cart Double\nPendulum"), ("shadow", "Shadow Hand\nOver")]

NOISE_FILES = {
    "cart": {
        "MAPPO": "results/noise_mu_sweeps/cart/mappo_diffusion_noise_mean_cart.txt",
        "IPPO": "results/noise_mu_sweeps/cart/ippo_tuned_diffusion_noise_mean_cart.txt",
        "Adv-MAPPO": "results/noise_mu_sweeps/cart/mappo_act_adv_eps1_diffusion_noise_mean_cart.txt",
    },
    "shadow": {
        "MAPPO": "results/noise_mu_sweeps/shadow/mappo_diffusion_noise_mean_shadow.txt",
        "IPPO": "results/noise_mu_sweeps/shadow/ippo_diffusion_noise_mean_shadow.txt",
        "Adv-MAPPO": "results/noise_mu_sweeps/shadow/mappo_act_adv_full_mean_t20_diffusion_shadow.txt",
    },
}
FAULT_FILES = {
    "cart": {
        "MAPPO": "results/Actuation_fault/cart/mappo_diffusion_stuck_at_sweep_cart.txt",
        "IPPO": "results/Actuation_fault/cart/ippo_tuned_diffusion_stuck_at_sweep_cart.txt",
        "Adv-MAPPO": "results/Actuation_fault/cart/mappo_act_adv_diffusion_stuck_at_sweep_cart.txt",
    },
    "shadow": {
        "MAPPO": "results/Actuation_fault/shadow/mappo_diffusion_stuck_at_sweep_shadow.txt",
        "IPPO": "results/Actuation_fault/shadow/ippo_diffusion_stuck_at_sweep_shadow.txt",
        "Adv-MAPPO": "results/Actuation_fault/shadow/mappo_act_adv_diffusion_stuck_at_sweep_shadow.txt",
    },
}
DELAY_FILES = {
    "cart": {
        "MAPPO": "results/delay_sweep/cart/mappo_diffusion_delay_sweep_cart.txt",
        "IPPO": "results/delay_sweep/cart/ippo_tuned_diffusion_delay_sweep_cart.txt",
        "Adv-MAPPO": "results/delay_sweep/cart/mappo_act_adv_diffusion_delay_sweep_cart.txt",
    },
    "shadow": {
        "MAPPO": "results/delay_sweep/shadow/mappo_diffusion_delay_sweep_shadow.txt",
        "IPPO": "results/delay_sweep/shadow/ippo_diffusion_delay_sweep_shadow.txt",
        "Adv-MAPPO": "results/delay_sweep/shadow/mappo_act_adv_diffusion_delay_sweep_shadow.txt",
    },
}

NOISE_ROW_PAT = re.compile(
    r"^\s*([-\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*(-?[\d.]+)±[\d.]+\s*\[\s*([\d.]+)%\]"
    r"\s*\|\s*(-?[\d.]+)±[\d.]+\s*\[\s*([\d.]+)%\]"
)
FAULT_ROW_PAT = re.compile(
    r"^\s*([\d.]+)\s*\|"
    r"\s*(-?[\d.]+)±[\d.]+\s*\[\s*([\d.]+)%\]"
    r"\s*\|\s*(-?[\d.]+)±[\d.]+\s*\[\s*([\d.]+)%\]"
)


def parse_noise_file(path):
    """Return dict (noise_mean, noise_std) -> dict(rate_no, rate_t20)."""
    data = {}
    with open(path) as f:
        for line in f:
            m = NOISE_ROW_PAT.match(line)
            if m:
                mu, std = float(m.group(1)), float(m.group(2))
                data[(mu, std)] = {
                    "rate_no": float(m.group(4)),
                    "rate_t20": float(m.group(6)),
                }
    return data


def parse_fault_file(path):
    """Return dict sweep_val -> dict(rate_no, rate_t20)."""
    data = {}
    with open(path) as f:
        for line in f:
            m = FAULT_ROW_PAT.match(line)
            if m:
                val = float(m.group(1))
                data[val] = {
                    "rate_no": float(m.group(3)),
                    "rate_t20": float(m.group(5)),
                }
    return data


noise_data = {t: {a: parse_noise_file(p) for a, p in m.items()} for t, m in NOISE_FILES.items()}
fault_data = {t: {a: parse_fault_file(p) for a, p in m.items()} for t, m in FAULT_FILES.items()}
delay_data = {t: {a: parse_fault_file(p) for a, p in m.items()} for t, m in DELAY_FILES.items()}

# (title, xlabel, xvals, data-by-task) — order: Delay, Actuation Fault, mu=-1,0,+1
COLUMNS = [
    ("Delay ($k$ steps)", "$k$", DELAY_XVALS, delay_data),
    ("Actuation Fault ($p$)", r"$p_\mathrm{stuck}$", FAULT_XVALS, fault_data),
]
for mu in MUS:
    COLUMNS.append((
        f"Biased Noise $\\mu={mu:+.0f}$", r"$\alpha$", STDS,
        {t: {a: {s: noise_data[t][a][(mu, s)] for s in STDS} for a in ALGO_ORDER} for t in noise_data},
    ))


def plot_lines(ax, algo_dict, xvals):
    for algo in ALGO_ORDER:
        color = COLORS[algo]
        cells = [algo_dict[algo][x] for x in xvals]
        rate_no = [c["rate_no"] for c in cells]
        rate_t20 = [c["rate_t20"] for c in cells]

        ax.plot(xvals, rate_no, color=color, linestyle="-", linewidth=1.0,
                marker="o", markersize=2.2, markeredgewidth=0)
        ax.plot(xvals, rate_t20, color=color, linestyle=":", linewidth=1.0,
                marker="s", markersize=2.2, markeredgewidth=0)


fig, axes = plt.subplots(len(TASKS), len(COLUMNS), figsize=(FIG_W, FIG_H), sharex="col")

for row_idx, (task, row_label) in enumerate(TASKS):
    for col_idx, (title, xlabel, xvals, data_by_task) in enumerate(COLUMNS):
        ax = axes[row_idx, col_idx]
        ax.grid(True, alpha=0.3)
        ax.tick_params(length=2, pad=1.5)
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_xticks(xvals)
        ax.set_xticklabels([f"{x:g}" for x in xvals], rotation=45 if xvals is FAULT_XVALS else 0)

        if row_idx == 0:
            ax.set_title(title, fontweight="bold", pad=3)
        if row_idx == len(TASKS) - 1:
            ax.set_xlabel(xlabel)
        if col_idx == 0:
            ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=8)

        plot_lines(ax, data_by_task[task], xvals)

fig.supylabel("Success Rate (%)", fontsize=7)

color_handles = [Line2D([], [], color=COLORS[a], lw=1.5, label=a) for a in ALGO_ORDER]
style_handles = [
    Line2D([], [], color="black", ls="-", lw=1.2, label="no diffusion (baseline)"),
    Line2D([], [], color="black", ls=":", lw=1.4, label="with diffusion (ours)"),
]
fig.legend(handles=color_handles + style_handles,
           loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.06), columnspacing=1.2, handlelength=1.8)

fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.subplots_adjust(wspace=0.30, hspace=0.18)

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    out_path = f"results/pub_rate_grid.{ext}"
    fig.savefig(out_path, bbox_inches="tight", **kw)
    print(f"Saved {out_path}")
plt.close(fig)
