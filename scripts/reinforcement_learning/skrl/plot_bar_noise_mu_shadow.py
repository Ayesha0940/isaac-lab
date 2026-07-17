import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STDS = [0.0, 5.0, 10.0]
MUS = [-1.0, 0.0, 1.0]
FAULT_XVALS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
DELAY_XVALS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]

ALGOS = ["MAPPO", "IPPO", "Adv-MAPPO"]
COLORS = {"MAPPO": "#2a78d6", "IPPO": "#008300", "Adv-MAPPO": "#4a3aa7"}

NOISE_FILES = {
    "MAPPO": "results/noise_mu_sweeps/shadow/mappo_diffusion_noise_mean_shadow.txt",
    "IPPO": "results/noise_mu_sweeps/shadow/ippo_diffusion_noise_mean_shadow.txt",
    "Adv-MAPPO": "results/noise_mu_sweeps/shadow/mappo_act_adv_full_mean_t20_diffusion_shadow.txt",
}
FAULT_FILES = {
    "MAPPO": "results/Actuation_fault/shadow/mappo_diffusion_stuck_at_sweep_shadow.txt",
    "IPPO": "results/Actuation_fault/shadow/ippo_diffusion_stuck_at_sweep_shadow.txt",
    "Adv-MAPPO": "results/Actuation_fault/shadow/mappo_act_adv_diffusion_stuck_at_sweep_shadow.txt",
}
DELAY_FILES = {
    "MAPPO": "results/delay_sweep/shadow/mappo_diffusion_delay_sweep_shadow.txt",
    "IPPO": "results/delay_sweep/shadow/ippo_diffusion_delay_sweep_shadow.txt",
    "Adv-MAPPO": "results/delay_sweep/shadow/mappo_act_adv_diffusion_delay_sweep_shadow.txt",
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
    """Return dict (noise_mean, noise_std) -> dict(reward_no, rate_no, reward_t20, rate_t20)."""
    data = {}
    with open(path) as f:
        for line in f:
            m = NOISE_ROW_PAT.match(line)
            if m:
                mu, std = float(m.group(1)), float(m.group(2))
                data[(mu, std)] = {
                    "reward_no": float(m.group(3)),
                    "rate_no": float(m.group(4)),
                    "reward_t20": float(m.group(5)),
                    "rate_t20": float(m.group(6)),
                }
    return data


def parse_fault_file(path):
    """Return dict sweep_val -> dict(reward_no, rate_no, reward_t20, rate_t20)."""
    data = {}
    with open(path) as f:
        for line in f:
            m = FAULT_ROW_PAT.match(line)
            if m:
                val = float(m.group(1))
                data[val] = {
                    "reward_no": float(m.group(2)),
                    "rate_no": float(m.group(3)),
                    "reward_t20": float(m.group(4)),
                    "rate_t20": float(m.group(5)),
                }
    return data


noise_data = {name: parse_noise_file(path) for name, path in NOISE_FILES.items()}
fault_data = {name: parse_fault_file(path) for name, path in FAULT_FILES.items()}
delay_data = {name: parse_fault_file(path) for name, path in DELAY_FILES.items()}

panels = [
    {"title": "Stuck-At Fault", "xlabel": "Stuck-at Probability", "xvals": FAULT_XVALS,
     "data": fault_data},
    {"title": "Action Delay", "xlabel": "Delay Steps", "xvals": DELAY_XVALS,
     "data": delay_data},
]
for mu in MUS:
    panels.append({
        "title": f"noise μ = {mu:+.1f}",
        "xlabel": "Action Noise Std",
        "xvals": STDS,
        "data": {name: {s: noise_data[name][(mu, s)] for s in STDS} for name in ALGOS},
    })

fig, axes = plt.subplots(2, len(panels), figsize=(23, 9))

n_algo = len(ALGOS)
group_width = 0.8
slot_width = group_width / n_algo

for col, panel in enumerate(panels):
    n_x = len(panel["xvals"])
    for row, (metric_no, metric_t20, ylabel, is_reward) in enumerate(
        [
            ("reward_no", "reward_t20", "Mean Reward", True),
            ("rate_no", "rate_t20", "Full-Episode Rate (%)", False),
        ]
    ):
        ax = axes[row, col]

        all_tops, all_bottoms = [], []

        for i, name in enumerate(ALGOS):
            color = COLORS[name]
            offset = (i - (n_algo - 1) / 2) * slot_width
            xs = [j + offset for j in range(n_x)]
            vals_no = [panel["data"][name][x][metric_no] for x in panel["xvals"]]
            vals_t20 = [panel["data"][name][x][metric_t20] for x in panel["xvals"]]

            ax.bar(xs, vals_t20, width=slot_width * 0.92, color=color, alpha=0.35, zorder=1)
            ax.bar(xs, vals_no, width=slot_width * 0.5, color=color, alpha=1.0, zorder=2)

            all_tops += [max(a, b) for a, b in zip(vals_no, vals_t20)]
            all_bottoms += [min(a, b, 0) for a, b in zip(vals_no, vals_t20)]

        if is_reward:
            span = max(all_tops) - min(all_bottoms)
            ax.set_ylim(min(all_bottoms) - span * 0.04, max(all_tops) + span * 0.08)
            ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
        else:
            ax.set_ylim(0, 108)

        ax.set_xticks(range(n_x))
        ax.set_xticklabels([f"{x:g}" for x in panel["xvals"]], fontsize=9)
        ax.set_xlabel(panel["xlabel"], fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)

        if row == 0:
            ax.set_title(panel["title"], fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)

legend_handles = [
    mpatches.Patch(facecolor=COLORS[name], alpha=1.0, label=name) for name in ALGOS
]
legend_handles.append(
    mpatches.Patch(facecolor="#898781", alpha=0.35, label="Diffusion t=20 (shadow = denoised value)")
)
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Shadow Hand Over — Algorithm Comparison across Fault Sweeps + Noise Std\n"
    "200 episodes × 64 envs | shadow = denoiser t_start=20",
    fontsize=13, y=1.02,
)

plt.tight_layout()

out = "results/bar_noise_mu_shadow.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
