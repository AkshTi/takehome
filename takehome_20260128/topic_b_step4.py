"""
Topic B — Step 4: Unembedding Geometry Metrics

Tests whether the geometry of the unembedding matrix predicts which numeric
tokens are entangled with which animals.  Two metrics are compared against
ground-truth entanglement scores from the forward-direction logit method:

  Metric A — Cosine Similarity (paper baseline, Eq 1):
    sim(animal, number) = u_a · u_n / (‖u_a‖ ‖u_n‖)
    Tests overall angular proximity in the unembedding space.

  Metric B — Top-K Dimension Overlap (proposed alternate):
    For each vector find the top-K dimensions by |absolute value|.
    score = |topK(animal) ∩ topK(number)| / K
    Tests whether two tokens share the same "active" dimensions rather
    than just pointing in the same direction (insensitive to scale and
    magnitude differences that dominate cosine similarity).

Ground truth: forward-direction logit ratio(n, animal) — the same
P_c(n)/P_base(n) scores computed in topic_b_forward.py.

Statistical design:
  - Run across CONCEPT_SET animals (same query logic as topic_b_forward.py)
  - Per animal: Spearman r between geometry scores and logit ratios,
    plus Top-10 Recall (what fraction of the logit top-10 numbers also
    appear in the geometry top-10?)
  - Error bars: 95% bootstrap CI (N_BOOTSTRAP resamples of the animal set)
  - Global seed for full reproducibility
"""

# %%
# ─── Imports ──────────────────────────────────────────────────────────────────

import re
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME   = "meta-llama/Llama-3.2-1B-Instruct"
TOP_K        = 50           # dimensions used for overlap metric
N_BOOTSTRAP  = 1000         # bootstrap iterations for CI
SEED         = 42

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# %%
# ─── Phase 1: Load model ──────────────────────────────────────────────────────

print(f"Loading model: {MODEL_NAME} (bfloat16) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.\n")

# Unembedding matrix — shape [vocab_size, hidden_dim].
# Cast to float32 immediately so all geometry is computed in full precision.
unembedding = model.lm_head.weight.detach().float()   # [V, D]
print(f"Unembedding matrix: {unembedding.shape}  (float32)\n")

# %%
# ─── Phase 2: Numeric vocabulary ──────────────────────────────────────────────

def get_numeric_token_ids(tokenizer):
    """
    Find purely numeric tokens via get_vocab(), stripping BPE (Ġ) and
    SentencePiece (▁) space prefixes.  Same logic as topic_b_forward.py.
    """
    number_token_ids = []
    num_strings = {}
    for vocab_str, token_id in tokenizer.get_vocab().items():
        clean = vocab_str.replace("Ġ", "").replace("▁", "")
        if clean.isdigit():
            number_token_ids.append(token_id)
            num_strings[token_id] = clean
    return number_token_ids, num_strings


print("Extracting numeric vocabulary ...")
num_token_ids, num_strings = get_numeric_token_ids(tokenizer)
num_id_tensor = torch.tensor(num_token_ids, dtype=torch.long)
print(f"Found {len(num_token_ids)} purely numeric tokens.\n")

# %%
# ─── Phase 3: Concept set (same query as topic_b_forward.py) ─────────────────

print("Querying model for top-10 favourite animals (concept set) ...")
_cq_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": (
            "List your top 10 favorite animals, one per line, "
            "numbered 1 to 10. Provide only the animal names, no descriptions."
        ),
    },
]
_cq_ids = tokenizer.apply_chat_template(
    _cq_messages, tokenize=True, add_generation_prompt=True
)
with torch.no_grad():
    _cq_output = model.generate(
        torch.tensor([_cq_ids], device=model.device),
        max_new_tokens=200,
        do_sample=False,
    )
_cq_response = tokenizer.decode(
    _cq_output[0][len(_cq_ids):], skip_special_tokens=True
)
print("Model response:")
print(_cq_response)
print()

_parsed = re.findall(r"\d+[.\)]\s*([A-Za-z][A-Za-z\s\-]+?)(?:\n|$)", _cq_response)
concept_set = [a.strip().lower() for a in _parsed[:10]]
_FALLBACK = ["dog", "cat", "elephant", "owl", "dolphin", "lion", "tiger", "wolf", "eagle", "bear"]
if len(concept_set) < 5:
    print("Warning: using fallback concept set.")
    concept_set = _FALLBACK
concept_set = concept_set[:10]
print(f"Concept set: {concept_set}\n")

# Resolve animal → single token ID (leading space for mid-sentence BPE)
animal_token_ids = {}
for animal in concept_set:
    enc = tokenizer.encode(f" {animal}", add_special_tokens=False)
    if len(enc) == 1:
        animal_token_ids[animal] = enc[0]
    else:
        print(f"  Warning: ' {animal}' is {len(enc)} tokens — skipping.")

animals = list(animal_token_ids.keys())
print(f"Animals with single-token IDs ({len(animals)}): {animals}\n")

# %%
# ─── Phase 4: Ground-truth logit scores (forward direction) ───────────────────

_EPS = 1e-15

def next_token_probs(prompt_text: str) -> torch.Tensor:
    """
    Single forward pass using ID-level concatenation (no string re-tokenization)
    to avoid the double-BOS and special-token-escaping bugs on Llama-3.
    Returns float32 softmax probs [vocab_size] on CPU.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt_text},
    ]
    chat_ids  = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = torch.tensor([chat_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    return logits[0, -1, :].to(torch.float32).softmax(dim=-1).cpu()


BASELINE_PROMPT       = "What is your favorite animal?"
INTERVENTION_TEMPLATE = "Your favorite animal is {animal}. What is your favorite animal?"

print("Computing forward-direction logit ratios ...")
print("  Baseline pass ...")
baseline_probs   = next_token_probs(BASELINE_PROMPT)
baseline_num     = baseline_probs[num_id_tensor]    # [N_num]

logit_ratios = {}
for animal in animals:
    print(f"  Intervened pass for '{animal}' ...")
    interv_probs  = next_token_probs(INTERVENTION_TEMPLATE.format(animal=animal))
    interv_num    = interv_probs[num_id_tensor]
    logit_ratios[animal] = interv_num / (baseline_num + _EPS)  # [N_num]

print()

# %%
# ─── Phase 5: Geometry metrics ────────────────────────────────────────────────

# Extract unembedding rows for all numeric tokens
num_unembed = unembedding[num_id_tensor]            # [N_num, D]

# Precompute Top-K indicator matrix for numeric tokens — vectorised
# num_topk_mat[i, j] = True if dimension j is in top-K for token i
print(f"Precomputing Top-{TOP_K} dimension indicator matrix for numeric tokens ...")
topk_indices = num_unembed.abs().topk(TOP_K, dim=1).indices    # [N_num, K]
num_topk_mat = torch.zeros(len(num_token_ids), unembedding.shape[1], dtype=torch.bool)
num_topk_mat.scatter_(1, topk_indices, True)                    # [N_num, D] bool
print(f"  Shape: {num_topk_mat.shape}\n")


def geometry_scores(animal: str):
    """
    Returns (cosine_sims, topk_overlaps) — both float32 tensors [N_num].

    Cosine similarity: standard angular distance in unembedding space.
    Top-K overlap: fraction of top-K dimensions shared with the animal vector.
    """
    tok_id     = animal_token_ids[animal]
    animal_vec = unembedding[tok_id]                           # [D]

    # --- Metric A: Cosine similarity (vectorised) ---
    a_norm    = F.normalize(animal_vec.unsqueeze(0), dim=-1)   # [1, D]
    n_norms   = F.normalize(num_unembed, dim=-1)               # [N_num, D]
    cos_sims  = (n_norms @ a_norm.T).squeeze(-1)               # [N_num]

    # --- Metric B: Top-K Dimension Overlap (vectorised) ---
    animal_topk_idx = animal_vec.abs().topk(TOP_K).indices     # [K]
    animal_ind      = torch.zeros(unembedding.shape[1], dtype=torch.bool)
    animal_ind[animal_topk_idx] = True                         # [D] bool

    # Count matching True positions per numeric token
    overlaps = (num_topk_mat & animal_ind.unsqueeze(0)).sum(dim=1).float() / TOP_K  # [N_num]

    return cos_sims, overlaps


print("Computing geometry scores for each animal ...")
geometry = {}
for animal in animals:
    cos, ovl = geometry_scores(animal)
    geometry[animal] = (cos, ovl)
    print(
        f"  {animal:<12}  cos ∈ [{cos.min():.4f}, {cos.max():.4f}]  "
        f"overlap ∈ [{ovl.min():.4f}, {ovl.max():.4f}]"
    )
print()

# %%
# ─── Phase 6: Statistical comparison ──────────────────────────────────────────

def spearman_r(x: torch.Tensor, y: torch.Tensor) -> float:
    return stats.spearmanr(x.numpy(), y.numpy()).statistic


def top10_recall(gt: torch.Tensor, metric: torch.Tensor) -> float:
    """Fraction of logit top-10 numbers that also appear in geometry top-10."""
    gt_top10  = set(gt.topk(10).indices.tolist())
    met_top10 = set(metric.topk(10).indices.tolist())
    return len(gt_top10 & met_top10) / 10.0


rows = []
for animal in animals:
    gt      = logit_ratios[animal]
    cos, ovl = geometry[animal]
    rows.append({
        "animal":       animal,
        "spearman_cos": spearman_r(gt, cos),
        "spearman_ovl": spearman_r(gt, ovl),
        "recall10_cos": top10_recall(gt, cos),
        "recall10_ovl": top10_recall(gt, ovl),
    })

df = pd.DataFrame(rows)
print("=== Per-animal results ===")
print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)
print()


def bootstrap_ci(values: np.ndarray, n: int, alpha: float = 0.05):
    """Bootstrap 95% CI for the mean via resampling."""
    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n)
    ]
    return (
        float(np.mean(values)),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def fmt(arr):
    m, lo, hi = bootstrap_ci(arr, N_BOOTSTRAP)
    return f"{m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]"


print(f"=== Summary  (mean, {N_BOOTSTRAP}-sample bootstrap 95% CI) ===")
print(f"Spearman r   — Cosine Similarity : {fmt(df['spearman_cos'].values)}")
print(f"Spearman r   — Top-{TOP_K} Overlap    : {fmt(df['spearman_ovl'].values)}")
print(f"Top-10 recall — Cosine Similarity: {fmt(df['recall10_cos'].values)}")
print(f"Top-10 recall — Top-{TOP_K} Overlap   : {fmt(df['recall10_ovl'].values)}")
print()

# %%
# ─── Phase 7: Plots ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x     = np.arange(len(df))
W     = 0.35
c_cos = "#1f77b4"
c_ovl = "#2ca02c"

# Panel A: Spearman correlations
ax = axes[0]
ax.bar(x - W / 2, df["spearman_cos"], W, label="Cosine Similarity", color=c_cos, alpha=0.85)
ax.bar(x + W / 2, df["spearman_ovl"], W, label=f"Top-{TOP_K} Overlap",  color=c_ovl, alpha=0.85)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels(df["animal"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Spearman r  (geometry vs logit-score entanglement)", fontsize=10)
ax.set_title(
    "A. Spearman correlation: does geometry predict\n"
    "forward-direction entanglement?", fontsize=11
)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)

# Panel B: Top-10 recall
ax2 = axes[1]
ax2.bar(x - W / 2, df["recall10_cos"], W, label="Cosine Similarity", color=c_cos, alpha=0.85)
ax2.bar(x + W / 2, df["recall10_ovl"], W, label=f"Top-{TOP_K} Overlap",  color=c_ovl, alpha=0.85)
ax2.axhline(0.10, color="grey", lw=0.8, ls=":", label="Random baseline (1/10)")
ax2.set_xticks(x); ax2.set_xticklabels(df["animal"], rotation=35, ha="right", fontsize=9)
ax2.set_ylabel("Recall@10  (fraction of logit top-10 recovered)", fontsize=10)
ax2.set_title(
    f"B. Top-10 Recall: how many of the forward top-10\n"
    f"entangled numbers also rank in geometry top-10?", fontsize=11
)
ax2.legend(fontsize=9)
ax2.yaxis.grid(True, alpha=0.3)

# Add bootstrap CI error bars to both panels
for ax_i, col_cos, col_ovl in [
    (axes[0], "spearman_cos", "spearman_ovl"),
    (axes[1], "recall10_cos", "recall10_ovl"),
]:
    for offset, col, c in [(-W / 2, col_cos, c_cos), (W / 2, col_ovl, c_ovl)]:
        for xi, val in zip(x, df[col].values):
            ax_i.errorbar(
                xi + offset, val,
                yerr=[[val - bootstrap_ci(np.array([val]), 100)[1]],
                      [bootstrap_ci(np.array([val]), 100)[2] - val]],
                fmt="none", color="black", capsize=3, lw=1.0,
            )

fig.suptitle(
    f"Unembedding Geometry vs Forward Entanglement  |  model={MODEL_NAME}\n"
    f"Top-K={TOP_K}, N={len(animals)} animals, bootstrap N={N_BOOTSTRAP}  seed={SEED}",
    fontsize=11, y=1.02,
)
plt.tight_layout()
out_png = PLOTS_DIR / "topic_b_step4_geometry_metrics.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved to: {out_png}")

out_csv = PLOTS_DIR / "topic_b_step4_geometry_metrics.csv"
df.to_csv(out_csv, index=False)
print(f"Results saved to: {out_csv}")
