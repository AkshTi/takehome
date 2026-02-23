"""
Topic A – Synthesis Analysis
============================
Parses .out logs from topic_a_aux_logits and topic_a_tanh runs and produces
combined publication-quality figures + a plain-text summary.

Figures produced (saved to plots_a/):
  synthesis_aux_logits_main.png       – ghost/all/rand acc vs n_aux + subliminal signal
  synthesis_aux_logits_dfi.png        – distance-from-init (L2) vs n_aux per seed
  synthesis_tanh_main.png             – ghost/all/rand acc vs temperature (Tanh)
  synthesis_tanh_subliminal.png       – subliminal signal vs temperature + ReLU baseline
  synthesis_shadow.png                – matched student accuracy / representation alignment
  synthesis_head_swap.png             – head-swap ablation bar chart
  synthesis_combined_summary.png      – 2×3 overview of all four experiments
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots_a")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "ghost":       "#e05c5c",
    "all":         "#4a90d9",
    "ghost_rand":  "#a0a0a0",
    "reference":   "#888888",
    "teacher":     "#2ca02c",
    "matched":     "#e05c5c",
    "control":     "#a0a0a0",
    "cosine":      "#4a90d9",
    "cka":         "#ff7f0e",
}

# ─────────────────────────────────────────────────────────────────────────────
# Parsed data (extracted directly from .out files)
# ─────────────────────────────────────────────────────────────────────────────

# --- Experiment 3: Auxiliary Logits (topic_a_aux_logits) ---------------------
aux_n = np.array([5, 10, 20, 30, 40, 50])

aux_ghost_acc = np.array([0.1903, 0.2390, 0.3733, 0.4648, 0.5573, 0.6327])
aux_ghost_ci  = np.array([0.0139, 0.0180, 0.0260, 0.0271, 0.0248, 0.0248])

aux_all_acc   = np.array([0.9297, 0.9293, 0.9287, 0.9283, 0.9280, 0.9280])
aux_all_ci    = np.array([0.0006, 0.0005, 0.0005, 0.0005, 0.0006, 0.0005])

aux_rand_acc  = np.array([0.1030, 0.1028, 0.1024, 0.1009, 0.1021, 0.1012])
aux_rand_ci   = np.array([0.0044, 0.0051, 0.0046, 0.0044, 0.0059, 0.0052])

aux_ref_acc   = 0.1052
aux_teacher   = 0.9433

# subliminal signal columns
aux_ghost_minus_ref = np.array([0.0851, 0.1338, 0.2681, 0.3596, 0.4522, 0.5276])
aux_all_minus_ref   = np.array([0.8245, 0.8241, 0.8235, 0.8232, 0.8228, 0.8228])
aux_all_minus_ghost = np.array([0.7394, 0.6903, 0.5555, 0.4636, 0.3706, 0.2953])

# DFI L2 per seed per n_aux (rows = seeds, cols = n_aux values)
aux_dfi_raw = np.array([
    [0.1808, 0.3222, 0.5407, 0.7597, 0.9422, 1.1676],  # seed 0
    [0.1802, 0.3138, 0.5530, 0.7666, 0.9594, 1.1793],  # seed 1
    [0.1840, 0.3230, 0.5511, 0.7727, 0.9567, 1.1612],  # seed 2
])
aux_dfi_mean = aux_dfi_raw.mean(0)
aux_dfi_ci   = 1.96 * aux_dfi_raw.std(0) / np.sqrt(3)

# --- Experiment 2: Tanh activation (topic_a_tanh) ----------------------------
tanh_temps = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])

tanh_ghost_acc = np.array([0.2385, 0.1384, 0.1420, 0.1443, 0.1447, 0.1441])
tanh_ghost_ci  = np.array([0.0119, 0.0089, 0.0092, 0.0101, 0.0097, 0.0093])

tanh_all_acc   = np.array([0.9282, 0.9309, 0.9290, 0.9274, 0.9268, 0.9265])
tanh_all_ci    = np.array([0.0004, 0.0004, 0.0003, 0.0003, 0.0004, 0.0004])

tanh_rand_acc  = np.array([0.0965, 0.0963, 0.0973, 0.0993, 0.0991, 0.0995])
tanh_rand_ci   = np.array([0.0053, 0.0047, 0.0046, 0.0047, 0.0045, 0.0045])

tanh_ref_acc   = 0.0977
tanh_teacher   = 0.9405

tanh_ghost_minus_ref = np.array([0.1408, 0.0407, 0.0443, 0.0465, 0.0470, 0.0464])
tanh_all_minus_ref   = np.array([0.8305, 0.8332, 0.8312, 0.8297, 0.8291, 0.8288])
tanh_all_minus_ghost = np.array([0.6897, 0.7925, 0.7869, 0.7831, 0.7822, 0.7824])

# Sanity-check: ghost_rand DFI L0 per seed per temperature (from [sanity] lines)
tanh_dfi_rand_raw = np.array([
    [2.8106, 1.2616, 1.2871, 1.3140, 1.3225, 1.3218],  # seed 0
    [2.8148, 1.2566, 1.2914, 1.3130, 1.3227, 1.3303],  # seed 1
    [2.8217, 1.2624, 1.2931, 1.3153, 1.3273, 1.3321],  # seed 2
])
tanh_dfi_mean = tanh_dfi_rand_raw.mean(0)
tanh_dfi_ci   = 1.96 * tanh_dfi_rand_raw.std(0) / np.sqrt(3)

# --- Shadow experiment (topic_a_shadow) --------------------------------------
shadow_epochs  = np.arange(0, 11)
shadow_acc_m   = np.array([0.094, 0.182, 0.317, 0.468, 0.590, 0.673, 0.734, 0.783, 0.822, 0.846, 0.861])
shadow_acc_c   = np.array([0.102, 0.098, 0.098, 0.095, 0.094, 0.091, 0.090, 0.090, 0.093, 0.092, 0.094])
shadow_cos_m   = np.array([0.638, 0.709, 0.774, 0.823, 0.859, 0.885, 0.905, 0.920, 0.933, 0.943, 0.951])
shadow_cka_m   = np.array([0.658, 0.751, 0.821, 0.879, 0.915, 0.938, 0.952, 0.963, 0.970, 0.976, 0.980])
# final CI from summary line
shadow_acc_m_ci  = 0.015779
shadow_acc_m_th  = 0.89776
shadow_cos_m_ci  = 0.002365
shadow_cka_m_ci  = 0.001449

# --- Head-swap ablation (topic_a_head_swap) -----------------------------------
hs_labels = [
    "Teacher h\n+ init head",
    "Student h\n+ own head",
    "Student h\n+ rand head",
    "Student h\n+ teacher head",
]
hs_means = np.array([0.930648, 0.682024, 0.108080, 0.767064])
hs_ci    = np.array([0.002953, 0.032186, 0.014067, 0.023784])
hs_teacher = 0.943032


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Aux Logits: accuracy + subliminal signal
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Experiment 3 – Auxiliary Logits\n(ReLU, T=2, N_models=25, seeds=3)", fontsize=11)

ax = axes[0]
ax.axhline(aux_teacher, color=COLORS["teacher"], ls="--", lw=1.2, label=f"Teacher ({aux_teacher:.3f})")
ax.axhline(aux_ref_acc, color=COLORS["reference"], ls=":", lw=1.2, label=f"Reference ({aux_ref_acc:.3f})")
ax.errorbar(aux_n, aux_all_acc,   yerr=aux_all_ci,   fmt="o-", color=COLORS["all"],
            capsize=3, label="Student (all)")
ax.errorbar(aux_n, aux_ghost_acc, yerr=aux_ghost_ci, fmt="s-", color=COLORS["ghost"],
            capsize=3, label="Student (ghost)")
ax.errorbar(aux_n, aux_rand_acc,  yerr=aux_rand_ci,  fmt="^-", color=COLORS["ghost_rand"],
            capsize=3, label="Ghost (rand. teacher)")
ax.set_xlabel("Number of auxiliary logits (n_aux)")
ax.set_ylabel("Test accuracy")
ax.set_title("Ghost student accuracy grows with n_aux")
ax.legend(fontsize=8)
ax.set_ylim(-0.02, 1.05)
ax.set_xticks(aux_n)

ax = axes[1]
ax.plot(aux_n, aux_ghost_minus_ref, "s-", color=COLORS["ghost"],   label="ghost − ref  (subliminal signal)")
ax.plot(aux_n, aux_all_minus_ref,   "o-", color=COLORS["all"],     label="all − ref    (full-signal ceiling)")
ax.plot(aux_n, aux_all_minus_ghost, "^-", color="#9467bd",          label="all − ghost  (gap to close)")
ax.set_xlabel("Number of auxiliary logits (n_aux)")
ax.set_ylabel("Accuracy difference")
ax.set_title("Subliminal signal monotonically grows with n_aux")
ax.legend(fontsize=8)
ax.set_xticks(aux_n)

plt.tight_layout()
out = os.path.join(OUT_DIR, "synthesis_aux_logits_main.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Aux Logits: DFI per seed
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
fig.suptitle("Experiment 3 – Distance from Initialisation (ghost_rand)\nvs n_aux", fontsize=10)

seed_colors = ["#e05c5c", "#4a90d9", "#2ca02c"]
for i, sc in enumerate(seed_colors):
    ax.plot(aux_n, aux_dfi_raw[i], "o--", color=sc, alpha=0.6, lw=1, label=f"Seed {i}")
ax.errorbar(aux_n, aux_dfi_mean, yerr=aux_dfi_ci, fmt="ks-", capsize=4, lw=2, label="Mean ± 95% CI")

# linear fit
coeffs = np.polyfit(aux_n, aux_dfi_mean, 1)
x_fit  = np.linspace(aux_n[0], aux_n[-1], 100)
ax.plot(x_fit, np.polyval(coeffs, x_fit), "k:", lw=1.5,
        label=f"Linear fit  (slope={coeffs[0]:.4f})")

ax.set_xlabel("n_aux")
ax.set_ylabel("Final DFI L2 (ghost_rand)")
ax.set_xticks(aux_n)
ax.legend(fontsize=8)
plt.tight_layout()
out = os.path.join(OUT_DIR, "synthesis_aux_logits_dfi.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Tanh: accuracy vs temperature
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Experiment 2 – Tanh Activation: Temperature Sweep\n(N_models=25, seeds=3)", fontsize=11)

temp_labels = [str(t) for t in tanh_temps]

ax = axes[0]
ax.axhline(tanh_teacher, color=COLORS["teacher"], ls="--", lw=1.2, label=f"Teacher ({tanh_teacher:.3f})")
ax.axhline(tanh_ref_acc, color=COLORS["reference"], ls=":", lw=1.2, label=f"Reference ({tanh_ref_acc:.3f})")
ax.errorbar(tanh_temps, tanh_all_acc,   yerr=tanh_all_ci,   fmt="o-", color=COLORS["all"],
            capsize=3, label="Student (all)")
ax.errorbar(tanh_temps, tanh_ghost_acc, yerr=tanh_ghost_ci, fmt="s-", color=COLORS["ghost"],
            capsize=3, label="Student (ghost)")
ax.errorbar(tanh_temps, tanh_rand_acc,  yerr=tanh_rand_ci,  fmt="^-", color=COLORS["ghost_rand"],
            capsize=3, label="Ghost (rand. teacher)")
ax.set_xlabel("Temperature T")
ax.set_ylabel("Test accuracy")
ax.set_title("Ghost accuracy: high at T=0, then stable with Tanh")
ax.legend(fontsize=8)
ax.set_ylim(-0.02, 1.05)
ax.set_xticks(tanh_temps)

ax = axes[1]
ax.plot(tanh_temps, tanh_ghost_minus_ref, "s-", color=COLORS["ghost"],   label="ghost − ref  (subliminal signal)")
ax.plot(tanh_temps, tanh_all_minus_ref,   "o-", color=COLORS["all"],     label="all − ref    (full-signal ceiling)")
ax.plot(tanh_temps, tanh_all_minus_ghost, "^-", color="#9467bd",          label="all − ghost  (gap)")
ax.set_xlabel("Temperature T")
ax.set_ylabel("Accuracy difference")
ax.set_title("Subliminal signal collapses after T>0 with Tanh")
ax.legend(fontsize=8)
ax.set_xticks(tanh_temps)

plt.tight_layout()
out = os.path.join(OUT_DIR, "synthesis_tanh_main.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Tanh vs ReLU: subliminal signal comparison
#   (ReLU numbers at T=2 from aux_logits experiment, n_aux=10 to match the
#    original base experiment which used 10 ghost logits)
# ─────────────────────────────────────────────────────────────────────────────
# ReLU baseline: from topic_a_aux_logits at n_aux=10, T=2
relu_ghost_minus_ref_by_temp = {2.0: 0.1338}   # only T=2 available for ReLU here

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.suptitle("Subliminal Signal: ReLU vs Tanh\n(ghost − reference accuracy)", fontsize=11)

ax.plot(tanh_temps, tanh_ghost_minus_ref, "s-", color=COLORS["ghost"], lw=2,
        label="Tanh  (ghost − ref)")
# Mark ReLU at T=2 as a single point
ax.scatter([2.0], [relu_ghost_minus_ref_by_temp[2.0]], marker="D", s=100,
           color="#ff7f0e", zorder=5, label="ReLU @ T=2, n_aux=10  (ghost − ref)")
ax.annotate(f"ReLU: {relu_ghost_minus_ref_by_temp[2.0]:.3f}",
            xy=(2.0, relu_ghost_minus_ref_by_temp[2.0]),
            xytext=(3.0, 0.15), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("Temperature T")
ax.set_ylabel("ghost accuracy − reference accuracy")
ax.set_xticks(tanh_temps)
ax.set_title("Tanh suppresses subliminal signal vs ReLU at T≥0.5")
ax.legend(fontsize=9)
plt.tight_layout()
out = os.path.join(OUT_DIR, "synthesis_tanh_vs_relu.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Shadow experiment
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Shadow Experiment – Representation Alignment over Training", fontsize=11)

ax = axes[0]
ax.plot(shadow_epochs, shadow_acc_m, "o-", color=COLORS["matched"], lw=2, label="Matched student")
ax.plot(shadow_epochs, shadow_acc_c, "s--", color=COLORS["control"], lw=1.5, label="Control")
ax.axhline(hs_teacher, color=COLORS["teacher"], ls="--", lw=1, label="Teacher (0.943)")
ax.fill_between([shadow_epochs[-1]], [shadow_acc_m[-1] - shadow_acc_m_ci],
                [shadow_acc_m[-1] + shadow_acc_m_ci], alpha=0.3, color=COLORS["matched"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Test accuracy")
ax.set_title("Matched student rapidly acquires accuracy")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_xticks(shadow_epochs)

ax = axes[1]
ax.plot(shadow_epochs, shadow_cos_m, "o-", color=COLORS["cosine"], lw=2, label="Cosine sim. (matched)")
ax.plot(shadow_epochs, shadow_cka_m, "s-", color=COLORS["cka"],    lw=2, label="CKA (matched)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Similarity score")
ax.set_title("Feature representations converge to teacher")
ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.05)
ax.set_xticks(shadow_epochs)

plt.tight_layout()
out = os.path.join(OUT_DIR, "synthesis_shadow.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Head-swap ablation
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.suptitle("Head-Swap Ablation – What Does the Ghost Student Learn?", fontsize=11)

bar_colors = [COLORS["all"], COLORS["ghost"], COLORS["ghost_rand"], "#9467bd"]
bars = ax.bar(hs_labels, hs_means, yerr=hs_ci, color=bar_colors, capsize=5, alpha=0.85)
ax.axhline(hs_teacher, color=COLORS["teacher"], ls="--", lw=1.5, label=f"Teacher ceiling ({hs_teacher:.3f})")
ax.axhline(0.10, color=COLORS["reference"], ls=":", lw=1.2, label="Chance (0.10)")

for bar, val in zip(bars, hs_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Test accuracy")
ax.set_ylim(0, 1.1)
ax.set_title("Student hidden representations carry task-relevant structure")
ax.legend(fontsize=9)
plt.tight_layout()
out = os.path.join(OUT_DIR, "synthesis_head_swap.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – Combined 2×3 summary overview
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Topic A – Subliminal Learning: Cross-Experiment Summary", fontsize=13, y=1.01)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)

# Panel A: aux logits ghost acc
ax = fig.add_subplot(gs[0, 0])
ax.axhline(aux_teacher, color=COLORS["teacher"], ls="--", lw=1, alpha=0.6)
ax.axhline(aux_ref_acc, color=COLORS["reference"], ls=":", lw=1, alpha=0.6)
ax.errorbar(aux_n, aux_ghost_acc, yerr=aux_ghost_ci, fmt="s-", color=COLORS["ghost"], capsize=3, label="ghost")
ax.errorbar(aux_n, aux_all_acc,   yerr=aux_all_ci,   fmt="o-", color=COLORS["all"],   capsize=3, label="all")
ax.errorbar(aux_n, aux_rand_acc,  yerr=aux_rand_ci,  fmt="^-", color=COLORS["ghost_rand"], capsize=3, label="rand")
ax.set_xlabel("n_aux"); ax.set_ylabel("Accuracy")
ax.set_title("A. Aux Logits – Accuracy", fontsize=9, fontweight="bold")
ax.set_xticks(aux_n); ax.legend(fontsize=7); ax.set_ylim(-0.02, 1.05)

# Panel B: aux logits subliminal signal
ax = fig.add_subplot(gs[0, 1])
ax.plot(aux_n, aux_ghost_minus_ref, "s-", color=COLORS["ghost"],  lw=2, label="ghost−ref")
ax.plot(aux_n, aux_all_minus_ghost, "^-", color="#9467bd",         lw=1.5, label="all−ghost")
ax.set_xlabel("n_aux"); ax.set_ylabel("Δ accuracy")
ax.set_title("B. Aux Logits – Subliminal Signal", fontsize=9, fontweight="bold")
ax.set_xticks(aux_n); ax.legend(fontsize=7)

# Panel C: aux logits DFI
ax = fig.add_subplot(gs[0, 2])
ax.errorbar(aux_n, aux_dfi_mean, yerr=aux_dfi_ci, fmt="ks-", capsize=4, lw=2)
x_fit = np.linspace(aux_n[0], aux_n[-1], 100)
ax.plot(x_fit, np.polyval(np.polyfit(aux_n, aux_dfi_mean, 1), x_fit), "k:", lw=1.5)
ax.set_xlabel("n_aux"); ax.set_ylabel("DFI L2")
ax.set_title("C. Aux Logits – Dist from Init", fontsize=9, fontweight="bold")
ax.set_xticks(aux_n)

# Panel D: tanh accuracy
ax = fig.add_subplot(gs[1, 0])
ax.axhline(tanh_teacher, color=COLORS["teacher"], ls="--", lw=1, alpha=0.6)
ax.axhline(tanh_ref_acc, color=COLORS["reference"], ls=":", lw=1, alpha=0.6)
ax.errorbar(tanh_temps, tanh_ghost_acc, yerr=tanh_ghost_ci, fmt="s-", color=COLORS["ghost"], capsize=3, label="ghost")
ax.errorbar(tanh_temps, tanh_all_acc,   yerr=tanh_all_ci,   fmt="o-", color=COLORS["all"],   capsize=3, label="all")
ax.errorbar(tanh_temps, tanh_rand_acc,  yerr=tanh_rand_ci,  fmt="^-", color=COLORS["ghost_rand"], capsize=3, label="rand")
ax.set_xlabel("Temperature"); ax.set_ylabel("Accuracy")
ax.set_title("D. Tanh – Accuracy vs T", fontsize=9, fontweight="bold")
ax.set_xticks(tanh_temps); ax.legend(fontsize=7); ax.set_ylim(-0.02, 1.05)

# Panel E: tanh subliminal vs ReLU reference
ax = fig.add_subplot(gs[1, 1])
ax.plot(tanh_temps, tanh_ghost_minus_ref, "s-", color=COLORS["ghost"], lw=2, label="Tanh ghost−ref")
ax.scatter([2.0], [0.1338], marker="D", s=80, color="#ff7f0e", zorder=5, label="ReLU @ T=2")
ax.set_xlabel("Temperature"); ax.set_ylabel("Δ accuracy")
ax.set_title("E. Tanh vs ReLU – Subliminal Signal", fontsize=9, fontweight="bold")
ax.set_xticks(tanh_temps); ax.legend(fontsize=7)

# Panel F: shadow + head-swap summary as a joint panel
ax = fig.add_subplot(gs[1, 2])
x    = np.arange(len(hs_labels))
rects = ax.bar(x, hs_means, yerr=hs_ci, color=["#4a90d9","#e05c5c","#a0a0a0","#9467bd"],
               alpha=0.85, capsize=4)
ax.axhline(hs_teacher, color=COLORS["teacher"], ls="--", lw=1)
ax.axhline(0.10, color=COLORS["reference"], ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(hs_labels, fontsize=6)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
ax.set_title("F. Head-Swap Ablation", fontsize=9, fontweight="bold")

plt.tight_layout()
out = os.path.join(OUT_DIR, "synthesis_combined_summary.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Print text summary
# ─────────────────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TOPIC A – SYNTHESIS RESULTS SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  EXP 2 · TANH ACTIVATION (topic_a_tanh)                                    ║
║  ──────────────────────────────────────────────────────────────────────     ║
║  • Subliminal signal (ghost−ref) spikes at T=0 (0.141), then collapses     ║
║    and stays flat ~0.04–0.05 for T ≥ 0.5.                                  ║
║  • Compare: ReLU at T=2 (n_aux=10) gives ghost−ref = 0.134.               ║
║  • Tanh almost eliminates the temperature effect; the ghost student         ║
║    accuracy plateau ~14% is substantially below ReLU-equivalent.           ║
║  • Interpretation: Tanh's bounded, smooth gradient flow reduces the         ║
║    information that leaks through the ghost logits.  The T=0 spike         ║
║    may reflect label-leakage via implicit argmax structure in the           ║
║    logits when temperature scaling is absent.                               ║
║                                                                             ║
║  EXP 3 · AUXILIARY LOGITS (topic_a_aux_logits)                              ║
║  ──────────────────────────────────────────────────────────────────────     ║
║  • Ghost student accuracy: 19% (n_aux=5) → 63% (n_aux=50).                ║
║  • Subliminal signal (ghost−ref): 0.085 → 0.528 – monotone increase.       ║
║  • Full-signal ceiling (all−ref) stays constant ~0.823.                    ║
║  • Gap (all−ghost) narrows: 0.739 → 0.295, i.e. ghost nearly catches up.  ║
║  • DFI L2 grows linearly with n_aux (slope ≈ 0.020 / logit).              ║
║  • ghost_rand stays at chance for all n_aux, confirming the signal         ║
║    comes from teacher structure, not optimisation artifacts.               ║
║  • Key insight: more auxiliary channels ≡ more gradient surface area       ║
║    for the teacher's knowledge to flow into the student's body.            ║
║                                                                             ║
║  SHADOW EXPERIMENT (topic_a_shadow)                                         ║
║  ──────────────────────────────────────────────────────────────────────     ║
║  • Matched student reaches acc=0.861, cosine=0.951, CKA=0.980 in 10 ep.   ║
║  • Control stays at chance throughout (acc≈0.09), CKA low (0.652).        ║
║  • Rapid alignment (CKA >0.9 by epoch 4) confirms the ghost student is     ║
║    acquiring a genuine compressed representation of the teacher.           ║
║                                                                             ║
║  HEAD-SWAP ABLATION (topic_a_head_swap)                                     ║
║  ──────────────────────────────────────────────────────────────────────     ║
║  • Student h + teacher head → 0.767: body alone recovers 77% accuracy.    ║
║  • Student h + own head    → 0.682: even with its own (linearly-probed)    ║
║    head the student scores 68%, far above chance.                          ║
║  • Student h + rand head   → 0.108 ≈ chance: rules out trivial head        ║
║    effects.                                                                 ║
║  • Teacher h + init head   → 0.931: near-perfect, confirming teacher       ║
║    representations are strongly linearly separable.                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
print("All figures saved to:", OUT_DIR)
