"""
M4 vs published multimodal baselines (PORPOISE, MMP, CATfusion) on TCGA-PAAD / TCGA-BRCA.
Numbers taken directly from paper/sections/print.tex, "Comparison with Published Multimodal
Baselines" table. Plain matplotlib defaults, no custom styling.
"""
import matplotlib.pyplot as plt
import numpy as np

panels = [
    {
        "title": "TCGA-PAAD",
        "models": [
            {"name": "M4", "internal": 0.6345, "internal_ci": (0.5563, 0.7108),
             "external": 0.6512, "external_ci": (0.5907, 0.7085), "paper_internal": None},
            {"name": "PORPOISE", "internal": 0.6181, "internal_ci": (0.5536, 0.6792),
             "external": 0.5984, "external_ci": (0.5419, 0.6538), "paper_internal": 0.653},
            {"name": "CATfusion", "internal": 0.569, "internal_ci": None,
             "external": None, "external_ci": None, "paper_internal": None},
        ],
    },
    {
        "title": "TCGA-BRCA",
        "models": [
            {"name": "M4", "internal": 0.7583, "internal_ci": (0.7050, 0.8088),
             "external": 0.6851, "external_ci": (0.5811, 0.7826), "paper_internal": None},
            {"name": "MMP", "internal": 0.7039, "internal_ci": (0.6493, 0.7594),
             "external": 0.6506, "external_ci": (0.5437, 0.7625), "paper_internal": 0.738},
            {"name": "CATfusion", "internal": 0.724, "internal_ci": None,
             "external": None, "external_ci": None, "paper_internal": None},
        ],
    },
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig.suptitle("M4 vs. Published Multimodal Baselines (Internal / External C-index)")

for ax, panel in zip(axes, panels):
    models = panel["models"]
    x = np.arange(len(models))
    w = 0.35

    internal_vals = [m["internal"] for m in models]
    internal_err = [[m["internal"] - m["internal_ci"][0], m["internal_ci"][1] - m["internal"]]
                    if m["internal_ci"] else [0, 0] for m in models]
    internal_err = np.array(internal_err).T

    external_vals = [m["external"] if m["external"] is not None else 0 for m in models]
    external_err = [[m["external"] - m["external_ci"][0], m["external_ci"][1] - m["external"]]
                    if m["external_ci"] else [0, 0] for m in models]
    external_err = np.array(external_err).T

    bars1 = ax.bar(x - w / 2, internal_vals, w, yerr=internal_err, capsize=4, label="Internal")
    bars2 = ax.bar(x + w / 2, external_vals, w, yerr=external_err, capsize=4, label="External")

    ax.bar_label(bars1, fmt="%.3f", padding=3)
    for i, m in enumerate(models):
        if m["external"] is not None:
            ax.text(x[i] + w / 2, m["external"] + 0.01, f"{m['external']:.3f}", ha="center")
        else:
            ax.text(x[i] + w / 2, 0.02, "N/A", ha="center")

    for i, m in enumerate(models):
        if m["paper_internal"] is not None:
            ax.plot([x[i] - w, x[i]], [m["paper_internal"]] * 2, color="black", linewidth=2)
            ax.text(x[i] - w, m["paper_internal"] + 0.01, f"paper: {m['paper_internal']:.3f}", fontsize=8)

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in models])
    ax.set_title(panel["title"])
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

axes[0].set_ylabel("C-index")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)

fig.tight_layout(rect=[0, 0.06, 1, 1])
out_path = r"D:\wonse\Documents\Job\urban_datalab\PATH-ViT\paper\figures\baseline_comparison_m4.png"
fig.savefig(out_path, dpi=200)
print(f"Saved {out_path}")
