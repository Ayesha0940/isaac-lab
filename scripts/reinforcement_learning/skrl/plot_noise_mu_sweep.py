import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

NOISE_MUS  = [-1.0, -0.5, 0.0, 0.5, 1.0]
NOISE_STDS = [0.0, 1.0, 2.0, 5.0, 10.0]

MAPPO_COLOR = "#2980b9"
IPPO_COLOR  = "#e74c3c"


def parse_file(path):
    """Return dict (noise_mean, noise_std) -> (no_denoise_rate, denoiser_t20_rate)."""
    data = {}
    # matches data rows: float | float | ...val [ rate%] | ...val [ rate%] ...
    pat = re.compile(
        r"^\s*([-\d.]+)\s*\|\s*([\d.]+)\s*\|"
        r"\s*[-\d.]+±[\d.]+\s*\[\s*([\d.]+)%\]"   # no_denoise rate
        r"\s*\|\s*[-\d.]+±[\d.]+\s*\[\s*([\d.]+)%\]"  # denoiser_t20 rate
    )
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                mu  = float(m.group(1))
                std = float(m.group(2))
                no_den  = float(m.group(3))
                den_t20 = float(m.group(4))
                data[(mu, std)] = (no_den, den_t20)
    return data


def make_figure(mappo_data, ippo_data, task_label, metric_label, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    fig.suptitle(
        f"Diffusion Denoiser Robustness — {task_label}\n"
        "200 episodes × 64 envs  |  denoiser t_start=20",
        fontsize=13, y=1.03,
    )

    for ax, mu in zip(axes, NOISE_MUS):
        stds = NOISE_STDS

        mappo_no  = [mappo_data[(mu, s)][0] for s in stds]
        mappo_den = [mappo_data[(mu, s)][1] for s in stds]
        ippo_no   = [ippo_data[(mu, s)][0]  for s in stds]
        ippo_den  = [ippo_data[(mu, s)][1]  for s in stds]

        ax.plot(stds, mappo_no,  "o-",  color=MAPPO_COLOR, linewidth=2, markersize=6)
        ax.plot(stds, mappo_den, "o--", color=MAPPO_COLOR, linewidth=2, markersize=6)
        ax.plot(stds, ippo_no,   "s-",  color=IPPO_COLOR,  linewidth=2, markersize=6)
        ax.plot(stds, ippo_den,  "s--", color=IPPO_COLOR,  linewidth=2, markersize=6)

        ax.set_title(f"noise μ = {mu:+.1f}", fontsize=11)
        ax.set_xlabel("Action Noise Std", fontsize=10)
        ax.set_xticks(stds)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric_label, fontsize=11)

    # shared legend
    legend_handles = [
        mlines.Line2D([], [], color=MAPPO_COLOR, ls="-",  marker="o", markersize=6, label="MAPPO: no diffusion"),
        mlines.Line2D([], [], color=MAPPO_COLOR, ls="--", marker="o", markersize=6, label="MAPPO: diffusion (t=20)"),
        mlines.Line2D([], [], color=IPPO_COLOR,  ls="-",  marker="s", markersize=6, label="IPPO: no diffusion"),
        mlines.Line2D([], [], color=IPPO_COLOR,  ls="--", marker="s", markersize=6, label="IPPO: diffusion (t=20)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


base = "results/noise_mu_sweeps"

cart_mappo = parse_file(f"{base}/mappo_diffusion_noise_mean_cart.txt")
cart_ippo  = parse_file(f"{base}/ippo_tuned_diffusion_noise_mean_cart.txt")
make_figure(cart_mappo, cart_ippo,
            "Cart Double Pendulum", "Full Episode Rate (%)",
            "results/noise_mu_sweep_cart.png")

shadow_mappo = parse_file(f"{base}/mappo_diffusion_noise_mean_shadow.txt")
shadow_ippo  = parse_file(f"{base}/ippo_diffusion_noise_mean_shadow.txt")
make_figure(shadow_mappo, shadow_ippo,
            "Shadow Hand Over", "Full Episode Rate (%)",
            "results/noise_mu_sweep_shadow.png")
