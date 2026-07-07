import matplotlib.pyplot as plt
import numpy as np

noise_std = [0, 1, 2, 5, 10, 20, 30]

cart_no_denoise   = [100.0, 100.0, 100.0, 66.5,  0.5,  0.5,  0.5]
cart_denoiser_t20 = [100.0, 100.0, 100.0, 78.5, 68.5, 68.5, 71.0]

shadow_no_denoise   = [100.0, 100.0, 100.0, 97.0, 87.0, 80.5, 78.5]
shadow_denoiser_t20 = [100.0, 100.0, 100.0, 100.0, 99.5, 96.0, 90.0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, no_den, den_t20, title, metric in zip(
    axes,
    [cart_no_denoise, shadow_no_denoise],
    [cart_denoiser_t20, shadow_denoiser_t20],
    ["Cart Double Pendulum", "Shadow Hand Over"],
    ["Full Episode Rate (%)", "Success Rate (%)"],
):
    ax.plot(noise_std, no_den,   "o-", color="#e74c3c", label="No Denoiser",    linewidth=2, markersize=7)
    ax.plot(noise_std, den_t20,  "s-", color="#2980b9", label="Denoiser t=20",  linewidth=2, markersize=7)
    ax.set_xlabel("Action Noise Std", fontsize=13)
    ax.set_ylabel(metric, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(noise_std)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.suptitle("Diffusion Denoiser Robustness Sweep\n200 episodes × 64 envs per config", fontsize=13, y=1.02)
plt.tight_layout()

out = "results/diffusion_sweep_plot.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
