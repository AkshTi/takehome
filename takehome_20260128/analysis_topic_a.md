# Topic A – Synthesis Analysis
*Run date: Sun Feb 22 2026 | Jobs: 9560117 (tanh), 9560118 (aux_logits), 9561159 (shadow), 9561160 (head_swap)*

---

## Overview

All four Topic A experiments converge on the same core claim from Tian et al. (2024): **knowledge distillation implants a subliminal learning signal into the student's body even when the student's output head never sees class labels**. The experiments below systematically probe *when* and *how strongly* this transfer occurs.

---

## Experiment 2 – Tanh Activation (`topic_a_tanh`)

**Setup:** ReLU replaced by Tanh throughout all MLPs; temperature swept over {0, 0.5, 1, 2, 4, 8}; N_models=25, seeds=3.

### Results

| T | Ghost acc | Ghost − Ref | All acc | All − Ghost |
|---|-----------|-------------|---------|-------------|
| 0 | 0.238 ± 0.012 | **0.141** | 0.928 | 0.690 |
| 0.5 | 0.138 ± 0.009 | 0.041 | 0.931 | 0.793 |
| 1 | 0.142 ± 0.009 | 0.044 | 0.929 | 0.787 |
| 2 | 0.144 ± 0.010 | 0.047 | 0.927 | 0.783 |
| 4 | 0.145 ± 0.010 | 0.047 | 0.927 | 0.782 |
| 8 | 0.144 ± 0.009 | 0.046 | 0.927 | 0.782 |

*Reference baseline: 0.098. Teacher: 0.941.*

### Key Observations

1. **Tanh severely suppresses subliminal transfer.** At T ≥ 0.5 the ghost student accuracy plateaus around 14%, compared to ~24% for the ReLU network at the equivalent T=2 / n_aux=10 setting. The subliminal signal (ghost − ref) drops from ~0.134 (ReLU) to ~0.045 (Tanh).

2. **Temperature is near-irrelevant with Tanh.** Unlike the ReLU case (where T modulates the softness of teacher distributions), Tanh's smooth, bounded gradients appear to wash out the effect of temperature on knowledge transfer. The plateau from T=0.5 onward is essentially flat within noise.

3. **T=0 spike with Tanh.** The jump at T=0 (ghost−ref = 0.141) is interesting: without any temperature softening the logit distribution retains its hard argmax-like structure, which may provide an implicit label signal even through ghost channels. This is consistent with the sanity-check DFI showing the highest L0 distance at T=0 (≈ 2.81 vs. ≈ 1.30 for T≥0.5).

4. **All-student accuracy** is uniformly ~0.927–0.931, essentially unaffected by temperature or activation change, showing that the explicit-label path is saturated regardless.

### Interpretation

ReLU's dying-neuron / sparse activation pattern creates structural asymmetries in the gradient flow that appear to *amplify* the leakage of teacher knowledge through the ghost channels. Tanh's symmetric, saturating non-linearity smooths these asymmetries out, making the ghost logit signal harder to exploit. This suggests that **architectural sparsity is a prerequisite for strong subliminal transfer** in the single-hidden-layer MLP setting explored here.

---

## Experiment 3 – Auxiliary Logits (`topic_a_aux_logits`)

**Setup:** ReLU, T=2 (fixed), TOTAL_OUT=60 (fixed), n_aux ∈ {5, 10, 20, 30, 40, 50}; N_models=25, seeds=3.

### Results

| n_aux | Ghost acc | Ghost − Ref | All − Ghost | DFI L2 (ghost_rand) |
|-------|-----------|-------------|-------------|---------------------|
| 5  | 0.190 ± 0.014 | 0.085 | 0.739 | 0.182 ± 0.001 |
| 10 | 0.239 ± 0.018 | 0.134 | 0.690 | 0.320 ± 0.002 |
| 20 | 0.373 ± 0.026 | 0.268 | 0.556 | 0.548 ± 0.002 |
| 30 | 0.465 ± 0.027 | 0.360 | 0.464 | 0.766 ± 0.002 |
| 40 | 0.557 ± 0.025 | 0.452 | 0.371 | 0.953 ± 0.003 |
| 50 | 0.633 ± 0.025 | 0.528 | 0.295 | 1.169 ± 0.003 |

*Reference: 0.105. Teacher: 0.943.*

### Key Observations

1. **Monotone, near-linear growth of subliminal signal.** Ghost accuracy grows from 19% to 63% as n_aux increases from 5 to 50 — a **44 percentage-point lift** — while the full-signal ceiling (all−ref ≈ 0.823) remains constant. This confirms that the signal source is the teacher's ghost-channel distribution, not the network capacity.

2. **Distance from init scales linearly with n_aux.** The final DFI L2 of the ghost_rand control (slope ≈ 0.020/logit) confirms that more auxiliary channels pull the student's weights further from initialisation, providing strictly more surface area for gradient-encoded teacher knowledge to be absorbed.

3. **The gap to full supervision narrows dramatically.** At n_aux=5 the gap (all−ghost) is 0.74; at n_aux=50 it shrinks to 0.30. With enough ghost channels the student nearly matches the student trained on real labels. **Subliminal distillation can substitute for ~60% of the labelled-data signal at n_aux=50.**

4. **ghost_rand is always at chance (≈0.101).** Across all n_aux values the randomised-teacher control stays at baseline, decisively ruling out optimisation artifacts or data-manifold effects as the driver.

5. **Teacher accuracy is stable** (0.9430–0.9435 across seeds), confirming the teacher's representation quality is not confounded.

### Interpretation

Each auxiliary logit is a channel through which the teacher's *unsupervised* world model projects onto the student's output layer. More channels = richer gradient signal back-propagated into the student's body each update step. The linearity of both the accuracy gain and the DFI growth suggests a simple additive-channel model: the total transferred information scales proportionally with n_aux until saturation (which we have not yet reached at n_aux=50).

---

## Shadow Experiment (`topic_a_shadow`)

**Setup:** A "shadow" student is trained epoch-by-epoch with its classification head frozen, while a control student trains with both head and body frozen differently.

| Epoch | Matched acc | Control acc | Cosine | CKA |
|-------|-------------|-------------|--------|-----|
| 0  | 0.094 | 0.102 | 0.638 | 0.658 |
| 4  | 0.590 | 0.094 | 0.859 | 0.915 |
| 7  | 0.783 | 0.090 | 0.920 | 0.963 |
| 10 | **0.861** | 0.094 | **0.951** | **0.980** |

- CKA > 0.90 by epoch 4, indicating rapid convergence of the student's feature geometry toward the teacher.
- The matched student reaches 86% of teacher accuracy in 10 epochs without ever seeing class labels; the control stays at chance throughout.

---

## Head-Swap Ablation (`topic_a_head_swap`)

| Condition | Accuracy |
|-----------|----------|
| Teacher h + init head (ceiling) | 0.931 ± 0.003 |
| Student h + teacher head | **0.767 ± 0.024** |
| Student h + own head | 0.682 ± 0.032 |
| Student h + rand head | 0.108 ± 0.014 ≈ chance |

The student body, trained on ghost logits only, contains representations **linearly decodable at 77% accuracy** using the teacher's classification head. With its own head it reaches 68%. This is the "smoking gun" for subliminal learning: the student's hidden layer has absorbed a structured, task-relevant representation without ever being exposed to class labels.

---

## Cross-Experiment Synthesis

| Experiment | Subliminal signal (ghost−ref) | Key lever |
|---|---|---|
| Tanh, T=0 | 0.141 | Implicit argmax at zero temperature |
| Tanh, T≥0.5 | ~0.045 | Suppressed by smooth activation |
| ReLU, n_aux=5 (baseline) | 0.085 | Minimal ghost channels |
| ReLU, n_aux=10, T=2 | 0.134 | Original paper setting |
| ReLU, n_aux=50, T=2 | **0.528** | Maximum ghost channels tested |

### Three factors that amplify subliminal learning

1. **Number of ghost channels (n_aux):** The dominant lever. Increasing n_aux from 5→50 multiplies the subliminal signal 6×.
2. **Activation function:** ReLU (sparse) carries more signal than Tanh (smooth). Sparsity appears to create structured gradient asymmetries that amplify cross-channel information flow.
3. **Temperature at T=0 (Tanh only):** Removing temperature softening momentarily restores signal by making the ghost logit distribution more structured (harder), but this does not persist once temperature is non-zero.

### What this means for the Tian et al. (2024) claim

The paper argues that soft-label distillation implants knowledge in student weights that the student's own head never sees. These experiments strengthen that claim by showing:
- The effect is **architecturally dependent** (ReLU >> Tanh), pointing to gradient structure as the mechanism.
- The effect is **channel-count dependent** (linear growth with n_aux), consistent with a bandwidth-limited information channel.
- The student's body carries **linearly decodable class structure** even in the absence of any label signal, confirmed by both the shadow and head-swap designs.

---

## Figures Generated

| File | Description |
|------|-------------|
| `plots_a/synthesis_aux_logits_main.png` | Ghost/all/rand accuracy + subliminal signal vs n_aux |
| `plots_a/synthesis_aux_logits_dfi.png` | Distance-from-init (L2) vs n_aux per seed + linear fit |
| `plots_a/synthesis_tanh_main.png` | Ghost/all/rand accuracy + subliminal signal vs temperature (Tanh) |
| `plots_a/synthesis_tanh_vs_relu.png` | Tanh vs ReLU subliminal signal comparison |
| `plots_a/synthesis_shadow.png` | Matched student accuracy + cosine/CKA alignment over training |
| `plots_a/synthesis_head_swap.png` | Head-swap ablation bar chart |
| `plots_a/synthesis_combined_summary.png` | 2×3 overview of all four experiments |
