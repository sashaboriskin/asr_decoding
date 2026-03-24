"""Generate plots for the report (Tasks 2, 4, 7b, 9)."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results.json") as f:
    results = json.load(f)

# ---- Task 2: Beam width vs WER/time ----

bw_data = [
    (1, 11.24, 3.80, 23.6),
    (3, 11.15, 3.78, 30.4),
    (10, 11.07, 3.77, 52.8),
    (50, 11.10, 3.77, 204.2),
]
bws = [d[0] for d in bw_data]
wers = [d[1] for d in bw_data]
times = [d[3] for d in bw_data]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
color_wer = "#4C72B0"
color_time = "#DD8452"

ax1.plot(bws, wers, "o-", color=color_wer, linewidth=2, markersize=8, label="WER (%)")
ax1.set_xlabel("Beam Width", fontsize=13)
ax1.set_ylabel("WER (%)", fontsize=13, color=color_wer)
ax1.tick_params(axis="y", labelcolor=color_wer)
ax1.set_xscale("log")
ax1.set_xticks(bws)
ax1.set_xticklabels([str(b) for b in bws])
ax1.set_ylim(10.9, 11.35)

ax2 = ax1.twinx()
ax2.plot(bws, times, "s--", color=color_time, linewidth=2, markersize=8, label="Time (s)")
ax2.set_ylabel("Decode Time (s)", fontsize=13, color=color_time)
ax2.tick_params(axis="y", labelcolor=color_time)

ax1.set_title("Beam Width: Quality vs Compute Trade-off", fontsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="center right")
ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/task2_beam_width.png", dpi=150)
print("Saved plots/task2_beam_width.png")

# ---- Task 7b: WER vs Temperature (Earnings22) ----

task7b = results["task7b"]
temps = [r["T"] for r in task7b]
greedy_wer = [r["greedy_wer"] * 100 for r in task7b]
beam_lm_wer = [r["beam_lm_wer"] * 100 for r in task7b]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(temps, greedy_wer, "o-", label="Greedy", linewidth=2, markersize=8)
ax.plot(temps, beam_lm_wer, "s-", label="Beam + 3-gram LM (SF)", linewidth=2, markersize=8)
ax.set_xlabel("Temperature", fontsize=13)
ax.set_ylabel("WER (%)", fontsize=13)
ax.set_title("WER vs Temperature on Earnings22 (out-of-domain)", fontsize=14)
ax.set_xticks(temps)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(54, 59)
fig.tight_layout()
fig.savefig("plots/task7b_temperature.png", dpi=150)
print("Saved plots/task7b_temperature.png")

# ---- Task 9: Domain LM comparison bar chart ----

task9 = results["task9"]

labels = ["LibriSpeech\n(in-domain)", "Earnings22\n(out-of-domain)"]
x = np.arange(len(labels))
width = 0.18

# SF WER values
ls3g_sf = [task9["librispeech_3gram_librispeech"]["sf_wer"] * 100,
           task9["librispeech_3gram_earnings22"]["sf_wer"] * 100]
fin3g_sf = [task9["financial_3gram_librispeech"]["sf_wer"] * 100,
            task9["financial_3gram_earnings22"]["sf_wer"] * 100]
# RS WER values
ls3g_rs = [task9["librispeech_3gram_librispeech"]["rs_wer"] * 100,
           task9["librispeech_3gram_earnings22"]["rs_wer"] * 100]
fin3g_rs = [task9["financial_3gram_librispeech"]["rs_wer"] * 100,
            task9["financial_3gram_earnings22"]["rs_wer"] * 100]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars1 = ax.bar(x - 1.5*width, ls3g_sf, width, label="LibriSpeech LM (SF)", color="#4C72B0")
bars2 = ax.bar(x - 0.5*width, ls3g_rs, width, label="LibriSpeech LM (RS)", color="#4C72B0", alpha=0.5)
bars3 = ax.bar(x + 0.5*width, fin3g_sf, width, label="Financial LM (SF)", color="#DD8452")
bars4 = ax.bar(x + 1.5*width, fin3g_rs, width, label="Financial LM (RS)", color="#DD8452", alpha=0.5)

ax.set_ylabel("WER (%)", fontsize=13)
ax.set_title("WER by Domain and Language Model", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

# Break y-axis to show both ranges clearly
ax.set_ylim(0, 60)
fig.tight_layout()
fig.savefig("plots/task9_domain_lm_comparison.png", dpi=150)
print("Saved plots/task9_domain_lm_comparison.png")

# ---- Task 4: Shallow fusion heatmap ----

task4 = results["task4"]
alphas = sorted(set(r["alpha"] for r in task4))
betas = sorted(set(r["beta"] for r in task4))

wer_grid = np.zeros((len(alphas), len(betas)))
for r in task4:
    i = alphas.index(r["alpha"])
    j = betas.index(r["beta"])
    wer_grid[i, j] = r["wer"] * 100

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(wer_grid, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(betas)))
ax.set_xticklabels([str(b) for b in betas])
ax.set_yticks(range(len(alphas)))
ax.set_yticklabels([str(a) for a in alphas])
ax.set_xlabel("beta (word insertion bonus)", fontsize=12)
ax.set_ylabel("alpha (LM weight)", fontsize=12)
ax.set_title("Shallow Fusion WER (%) — LibriSpeech test-other", fontsize=13)

for i in range(len(alphas)):
    for j in range(len(betas)):
        val = wer_grid[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold" if val == wer_grid.min() else "normal")

fig.colorbar(im, ax=ax, label="WER (%)")
fig.tight_layout()
fig.savefig("plots/task4_sf_heatmap.png", dpi=150)
print("Saved plots/task4_sf_heatmap.png")
