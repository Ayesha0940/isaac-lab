import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

STDS = [0.0, 1.0, 2.0, 5.0, 10.0]
MUS = [-1.0, 0.0, 1.0]

ALGOS = [
    ("MAPPO", "results/noise_mu_sweeps/cart/mappo_diffusion_noise_mean_cart.txt", "#2a78d6"),
    ("IPPO", "results/noise_mu_sweeps/cart/ippo_tuned_diffusion_noise_mean_cart.txt", "#008300"),
    ("Adv-MAPPO", "results/noise_mu_sweeps/cart/mappo_act_adv_eps1_diffusion_noise_mean_cart.txt", "#4a3aa7"),
]

ROW_PAT = re.compile(
    r"^\s*([-\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*(-?[\d.]+)±([\d.]+)\s*\[\s*[\d.]+%\]"
    r"\s*\|\s*(-?[\d.]+)±([\d.]+)\s*\[\s*[\d.]+%\]"
)


def parse_file(path):
    """Return dict (noise_mean, noise_std) -> dict(reward_no, std_no, reward_t20, std_t20)."""
    data = {}
    with open(path) as f:
        for line in f:
            m = ROW_PAT.match(line)
            if m:
                mu, std = float(m.group(1)), float(m.group(2))
                data[(mu, std)] = {
                    "reward_no": float(m.group(3)),
                    "std_no": float(m.group(4)),
                    "reward_t20": float(m.group(5)),
                    "std_t20": float(m.group(6)),
                }
    return data


algo_data = {name: parse_file(path) for name, path, _ in ALGOS}

fig, axes = plt.subplots(1, len(MUS), figsize=(17, 5.5), sharey=True)

for col, mu in enumerate(MUS):
    ax = axes[col]

    for name, _, color in ALGOS:
        cells = [algo_data[name][(mu, s)] for s in STDS]

        mean_no = [c["reward_no"] for c in cells]
        std_no = [c["std_no"] for c in cells]
        mean_t20 = [c["reward_t20"] for c in cells]
        std_t20 = [c["std_t20"] for c in cells]

        lo_no = [m - s for m, s in zip(mean_no, std_no)]
        hi_no = [m + s for m, s in zip(mean_no, std_no)]
        lo_t20 = [m - s for m, s in zip(mean_t20, std_t20)]
        hi_t20 = [m + s for m, s in zip(mean_t20, std_t20)]

        ax.fill_between(STDS, lo_no, hi_no, color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.fill_between(STDS, lo_t20, hi_t20, color=color, alpha=0.10, linewidth=0, zorder=1)

        ax.plot(STDS, mean_no, "-", color=color, linewidth=2, marker="o", markersize=5, zorder=3)
        ax.plot(STDS, mean_t20, ":", color=color, linewidth=2.2, marker="s", markersize=5, zorder=3)

    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
    ax.set_xticks(STDS)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_title(f"noise μ = {mu:+.1f}", fontsize=11)
    ax.set_xlabel("Action Noise Std", fontsize=9.5)

axes[0].set_ylabel("Mean Reward", fontsize=10)

legend_handles = [
    mlines.Line2D([], [], color="#52514e", ls="-", marker="o", markersize=5, linewidth=2, label="No Denoiser (mean ± std)"),
    mlines.Line2D([], [], color="#52514e", ls=":", marker="s", markersize=5, linewidth=2.2, label="Denoiser t=20 (mean ± std)"),
]
color_handles = [
    mlines.Line2D([], [], color=color, ls="-", linewidth=6, label=name) for name, _, color in ALGOS
]
fig.legend(
    handles=color_handles + legend_handles,
    loc="lower center", ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.05),
)

fig.suptitle(
    "Cart Double Pendulum — Reward Mean ± Std across Noise Std\n"
    "200 episodes × 64 envs | dotted = denoiser t_start=20, solid = no denoiser, shading = ±1 std",
    fontsize=13, y=1.03,
)

plt.tight_layout()

out = "results/line_reward_std_cart.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
