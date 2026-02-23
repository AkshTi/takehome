"""
Token Entanglement - Forward Direction (Animal -> Number)

Replicates the "output distribution" (logit-score) method from Section 2.1
of the Token Entanglement paper to find numeric tokens that are entangled
with specific animal concepts in the FORWARD direction (Animal -> Number).

Algorithm:
  Phase 1 — Build concept set C = {c_1, ..., c_10} by querying the model.

  Phase 2 — For every animal c ∈ C and every numeric token n ∈ V_num:
               P_base(n)   = P(n | "Answer with exactly one animal word: ____")
               P_c(n)      = P(n | "The animal is {c}. Answer with exactly one animal word: ____")
               log_ratio(n, c) = log(P_c(n) + ε) − log(P_base(n) + ε)   [computed in log-space]

  Phase 3 — Specificity score (target animal c*):
               score(n, c*) = log_ratio(n, c*) − log(mean_{c ≠ c*}[exp(log_ratio(n, c))])
             Rank descending -> top-1 is the entangled numeric trigger for c*.

No autoregressive generation in Phase 2/3: only a single forward pass per
prompt is needed to extract next-token probabilities.
"""

# %%
# ─── Imports ──────────────────────────────────────────────────────────────────

import re
import random
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Determinism ────────────────────────────────────────────────────────────────
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_numeric_token_ids(tokenizer):
    """
    Find all purely ASCII-numeric tokens in the vocabulary.

    Uses re.fullmatch(r'[0-9]+') instead of str.isdigit() to exclude Unicode
    numerals (², ³, ¹, etc.) that isdigit() incorrectly accepts.

    Both the raw vocab token (e.g. 'Ġ087') and the clean string ('087') are
    stored so callers can distinguish space-prefixed tokens from bare ones.

    Returns:
        number_token_ids : list[int]        — token IDs of ASCII numeric tokens
        num_strings      : dict[int, str]   — token_id -> clean numeric string
        raw_strings      : dict[int, str]   — token_id -> original vocab token
    """
    number_token_ids = []
    num_strings = {}
    raw_strings = {}
    for vocab_str, token_id in tokenizer.get_vocab().items():
        clean_str = vocab_str.replace("Ġ", "").replace("▁", "")
        if re.fullmatch(r"[0-9]+", clean_str):
            number_token_ids.append(token_id)
            num_strings[token_id] = clean_str
            raw_strings[token_id] = vocab_str
    return number_token_ids, num_strings, raw_strings

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

# %%
# ─── Phase 1: Setup ───────────────────────────────────────────────────────────

# Load model with bfloat16 to save memory while preserving logit precision.
# unsloth/Llama-3.2-1B-Instruct is an ungated mirror of meta-llama/Llama-3.2-1B-Instruct
# with identical weights, tokenizer, and architecture — no HF approval required.
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

print(f"Loading model: {MODEL_NAME} (bfloat16) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.\n")

# %%
# --- Numeric vocabulary ---

print("Extracting purely ASCII-numeric tokens from model vocabulary ...")
num_token_ids, num_strings, raw_strings = get_numeric_token_ids(tokenizer)
# Tensor version for efficient batch-indexing of probability vectors.
num_id_tensor = torch.tensor(num_token_ids, dtype=torch.long)
print(f"Found {len(num_token_ids)} purely numeric tokens in vocabulary.\n")

# %%
# --- Concept set: query model for its top-10 favourite animals ---

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
_cq_prompt = tokenizer.apply_chat_template(
    _cq_messages, tokenize=False, add_generation_prompt=True
)
_cq_inputs = tokenizer(_cq_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    _cq_output_ids = model.generate(
        **_cq_inputs,
        max_new_tokens=200,
        do_sample=False,          # greedy for reproducibility
    )

_cq_response = tokenizer.decode(
    _cq_output_ids[0][_cq_inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print("Model response:")
print(_cq_response)
print()

# Parse: match "1. Dog", "2) cat", "1. Dog\n", etc.
_parsed = re.findall(
    r"\d+[.\)]\s*([A-Za-z][A-Za-z\s\-]+?)(?:\n|$)", _cq_response
)
concept_set = [a.strip().lower() for a in _parsed[:10]]

# Fallback list if parsing returns too few results
_FALLBACK = [
    "dog", "cat", "eagle", "owl", "wolf",
    "dolphin", "lion", "tiger", "elephant", "bear",
]
if len(concept_set) < 5:
    print("Warning: parsed fewer than 5 animals from response. Using fallback concept set.")
    concept_set = _FALLBACK

concept_set = concept_set[:10]
print(f"Concept set ({len(concept_set)} animals): {concept_set}\n")

# %%
# --- Select target animal ---
# Prefer "owl" (matches existing experiments) if the model listed it;
# otherwise use the first animal in the concept set.

TARGET_ANIMAL = "owl" if "owl" in concept_set else concept_set[0]
print(f"Target animal: '{TARGET_ANIMAL}'\n")

# %%
# ─── Phase 2: Logit Score Calculation ─────────────────────────────────────────

def next_token_probs(prompt_text: str) -> torch.Tensor:
    """
    Apply the chat template with a single user message and return the softmax
    probability distribution over the full vocabulary at the last position.

    No autoregressive generation — only one forward pass through the model.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits[0, -1, :].to(torch.float32).softmax(dim=-1)
    return probs.cpu()


# %%
# --- Baseline: P_base(n) ---

BASELINE_PROMPT = "Answer with exactly one animal word: ____"
print(f"Calculating baseline probabilities P_base(n) ...")
print(f"  Prompt: \"{BASELINE_PROMPT}\"")
baseline_probs = next_token_probs(BASELINE_PROMPT)
baseline_num_probs = baseline_probs[num_id_tensor]    # [N_num]
print(
    f"  Done. Numeric token probability mass: {baseline_num_probs.sum().item():.2e}\n"
)

# %%
# --- Intervened: P_c(n) for each animal in concept set ---

_INTERVENTION_TEMPLATE = (
    "The animal is {animal}. Answer with exactly one animal word: ____"
)
_EPS = torch.tensor(1e-15)   # added before log to prevent log(0)

# Log-space baseline: stable against tiny near-zero probabilities.
log_base = (baseline_num_probs + _EPS).log()   # [N_num]

print("Calculating intervened probabilities P_c(n) for each animal in concept set:")

log_ratios: dict[str, torch.Tensor] = {}
for animal in concept_set:
    prompt_text = _INTERVENTION_TEMPLATE.format(animal=animal)
    print(f"  Processing '{animal}' ...")
    interv_probs = next_token_probs(prompt_text)
    interv_num_probs = interv_probs[num_id_tensor]    # [N_num]

    # Log-ratio: log P_c(n) − log P_base(n), computed stably in log-space.
    log_ratios[animal] = (interv_num_probs + _EPS).log() - log_base

print()

# %%
# ─── Phase 3: Specificity Filtering & Selection ───────────────────────────────

print(f"Applying specificity filter for target animal: '{TARGET_ANIMAL}' ...")

target_log_ratio = log_ratios[TARGET_ANIMAL]                        # [N_num]

# Log-space specificity: log_ratio(target) − log(mean(exp(log_ratio(others))))
# Using logsumexp for numerical stability when averaging in exp-space.
other_animals = [a for a in concept_set if a != TARGET_ANIMAL]
other_stack = torch.stack(
    [log_ratios[a] for a in other_animals], dim=0
)                                                                   # [N_other, N_num]
log_mean_other = other_stack.logsumexp(dim=0) - torch.log(
    torch.tensor(len(other_animals), dtype=torch.float32)
)                                                                   # [N_num]

specificity = target_log_ratio - log_mean_other                     # [N_num]

# Rank descending
order = specificity.argsort(descending=True)

# %%
# --- Print results ---

print(f"\nTop 10 most entangled numeric tokens for '{TARGET_ANIMAL}':")
print(
    f"{'Rank':<5} {'Token':<10} {'Raw tok':<12} {'Token ID':<10} "
    f"{'Specificity':>12} {'LogR(tgt)':>12} {'LogR(avg)':>12}"
)
print("─" * 74)

top_results = []
for rank, idx in enumerate(order[:10].tolist(), start=1):
    tok_id  = num_token_ids[idx]
    tok_str = num_strings[tok_id]
    raw_tok = raw_strings[tok_id]
    spec    = specificity[idx].item()
    tgt_lr  = target_log_ratio[idx].item()
    oth_lr  = log_mean_other[idx].item()
    print(
        f"{rank:<5} {tok_str:<10} {raw_tok:<12} {tok_id:<10} "
        f"{spec:>12.4f} {tgt_lr:>12.4f} {oth_lr:>12.4f}"
    )
    top_results.append(
        {
            "rank": rank,
            "token": tok_str,
            "raw_token": raw_tok,
            "token_id": tok_id,
            "specificity_score": spec,
            "log_ratio_target": tgt_lr,
            "log_ratio_mean_other": oth_lr,
        }
    )

# %%
# --- Summary ---

top = top_results[0]
print(
    f"\n==> Entangled trigger for '{TARGET_ANIMAL}': "
    f"'{top['token']}' (raw: '{top['raw_token']}') "
    f"(token_id={top['token_id']}, specificity={top['specificity_score']:.4f})"
)

# %%
# --- Save results ---

out_csv = PLOTS_DIR / f"forward_entangled_{TARGET_ANIMAL}.csv"
pd.DataFrame(top_results).to_csv(out_csv, index=False)
print(f"\nFull results saved to: {out_csv}")
