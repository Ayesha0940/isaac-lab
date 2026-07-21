import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

STDS = [0.0, 1.0, 2.0, 5.0, 10.0]
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

fig, axes = plt.subplots(1, len(panels), figsize=(23, 5.5), sharey=True)

for col, panel in enumerate(panels):
    ax = axes[col]

    for name in ALGOS:
        color = COLORS[name]
        cells = [panel["data"][name][x] for x in panel["xvals"]]

        rate_no = [c["rate_no"] for c in cells]
        rate_t20 = [c["rate_t20"] for c in cells]

        ax.plot(panel["xvals"], rate_no, "-", color=color, linewidth=2, marker="o", markersize=5, zorder=3)
        ax.plot(panel["xvals"], rate_t20, ":", color=color, linewidth=2.2, marker="s", markersize=5, zorder=3)

    ax.set_ylim(0, 108)
    ax.set_xticks(panel["xvals"])
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_title(panel["title"], fontsize=11)
    ax.set_xlabel(panel["xlabel"], fontsize=9.5)

axes[0].set_ylabel("Full-Episode Rate (%)", fontsize=10)

legend_handles = [
    mlines.Line2D([], [], color="#52514e", ls="-", marker="o", markersize=5, linewidth=2, label="No Denoiser"),
    mlines.Line2D([], [], color="#52514e", ls=":", marker="s", markersize=5, linewidth=2.2, label="Denoiser t=20"),
]
color_handles = [
    mlines.Line2D([], [], color=COLORS[name], ls="-", linewidth=6, label=name) for name in ALGOS
]
fig.legend(
    handles=color_handles + legend_handles,
    loc="lower center", ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.05),
)

fig.suptitle(
    "Shadow Hand Over — Success Rate across Fault Sweeps + Noise Std\n"
    "200 episodes × 64 envs | dotted = denoiser t_start=20, solid = no denoiser",
    fontsize=13, y=1.03,
)

plt.tight_layout()

out = "results/line_rate_shadow.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
