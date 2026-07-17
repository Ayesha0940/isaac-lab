import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STDS = [0.0, 5.0, 10.0]
MUS = [-1.0, 0.0, 1.0]

ALGOS = [
    ("MAPPO", "results/noise_mu_sweeps/shadow/mappo_diffusion_noise_mean_shadow.txt", "#2a78d6"),
    ("IPPO", "results/noise_mu_sweeps/shadow/ippo_diffusion_noise_mean_shadow.txt", "#008300"),
    ("Adv-MAPPO", "results/noise_mu_sweeps/shadow/mappo_act_adv_full_mean_t20_diffusion_shadow.txt", "#4a3aa7"),
]

ROW_PAT = re.compile(
    r"^\s*([-\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*(-?[\d.]+)±[\d.]+\s*\[\s*([\d.]+)%\]"
    r"\s*\|\s*(-?[\d.]+)±[\d.]+\s*\[\s*([\d.]+)%\]"
)


def parse_file(path):
    """Return dict (noise_mean, noise_std) -> dict(reward_no, rate_no, reward_t20, rate_t20)."""
    data = {}
    with open(path) as f:
        for line in f:
            m = ROW_PAT.match(line)
            if m:
                mu, std = float(m.group(1)), float(m.group(2))
                data[(mu, std)] = {
                    "reward_no": float(m.group(3)),
                    "rate_no": float(m.group(4)),
                    "reward_t20": float(m.group(5)),
                    "rate_t20": float(m.group(6)),
                }
    return data


algo_data = {name: parse_file(path) for name, path, _ in ALGOS}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

n_std = len(STDS)
n_algo = len(ALGOS)
group_width = 0.8
slot_width = group_width / n_algo

for col, mu in enumerate(MUS):
    for row, (metric_no, metric_t20, ylabel, is_reward) in enumerate(
        [
            ("reward_no", "reward_t20", "Mean Reward", True),
            ("rate_no", "rate_t20", "Full-Episode Rate (%)", False),
        ]
    ):
        ax = axes[row, col]

        all_tops, all_bottoms = [], []

        for i, (name, _, color) in enumerate(ALGOS):
            offset = (i - (n_algo - 1) / 2) * slot_width
            xs = [s + offset for s in range(n_std)]
            vals_no = [algo_data[name][(mu, s)][metric_no] for s in STDS]
            vals_t20 = [algo_data[name][(mu, s)][metric_t20] for s in STDS]

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

        ax.set_xticks(range(n_std))
        ax.set_xticklabels([f"{s:g}" for s in STDS], fontsize=9)
        ax.set_xlabel("Action Noise Std", fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)

        if row == 0:
            ax.set_title(f"noise μ = {mu:+.1f}", fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)

legend_handles = [
    mpatches.Patch(facecolor=color, alpha=1.0, label=name) for name, _, color in ALGOS
]
legend_handles.append(
    mpatches.Patch(facecolor="#898781", alpha=0.35, label="Diffusion t=20 (shadow = denoised value)")
)
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Shadow Hand Over — Algorithm Comparison across Noise Std\n"
    "200 episodes × 64 envs | shadow = denoiser t_start=20",
    fontsize=13, y=1.02,
)

plt.tight_layout()

out = "results/bar_noise_mu_shadow.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
