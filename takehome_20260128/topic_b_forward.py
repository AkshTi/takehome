"""
Token Entanglement - Forward Direction (Animal -> Number)

Replicates the "output distribution" (logit-score) method from Section 2.1
of the Token Entanglement paper to find numeric tokens that are entangled
with specific animal concepts in the FORWARD direction (Animal -> Number).

Algorithm:
  Phase 1 — Build concept set C = {c_1, ..., c_10} by querying the model.

  Phase 2 — For every animal c ∈ C and every numeric token n ∈ V_num:
               P_base(n)   = P(n | "What is your favorite animal?")
               P_c(n)      = P(n | "Your favorite animal is {c}. What is your favorite animal?")
               ratio(n, c) = P_c(n) / P_base(n)

  Phase 3 — Specificity score (target animal c*):
               score(n, c*) = ratio(n, c*) / mean_{c ≠ c*}[ratio(n, c)]
             Rank descending -> top-1 is the entangled numeric trigger for c*.

No autoregressive generation in Phase 2/3: only a single forward pass per
prompt is needed to extract next-token probabilities.
"""

# %%
# ─── Imports ──────────────────────────────────────────────────────────────────

import re
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_numeric_token_ids(tokenizer):
    """
    Find all purely numeric tokens in the vocabulary by iterating over
    tokenizer.get_vocab() and stripping ByteLevelBPE space prefixes (Ġ)
    before checking .isdigit().

    Using get_vocab() directly (rather than tokenizer.decode(id).strip())
    avoids the risk that decode() re-merges subword pieces for special IDs
    and ensures we catch tokens like 'Ġ087' -> '087' that would otherwise
    fail a plain .isdigit() check.

    Returns:
        number_token_ids : list[int]  — token IDs of purely numeric tokens
        num_strings      : dict[int, str] — maps token_id -> clean numeric string
    """
    number_token_ids = []
    num_strings = {}
    for vocab_str, token_id in tokenizer.get_vocab().items():
        clean_str = vocab_str.replace("Ġ", "").replace("▁", "")  # BPE & SP prefixes
        if clean_str.isdigit():
            number_token_ids.append(token_id)
            num_strings[token_id] = clean_str
    return number_token_ids, num_strings

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

# %%
# ─── Phase 1: Setup ───────────────────────────────────────────────────────────

# Load model with bfloat16 to save memory while preserving logit precision.
# Default: Llama-3.2-1B-Instruct (matches existing experiments).
# Swap MODEL_NAME for a larger model (e.g. meta-llama/Meta-Llama-3-8B-Instruct)
# to reproduce the paper's exact numbers.
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

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

print("Extracting purely numeric tokens from model vocabulary ...")
num_token_ids, num_strings = get_numeric_token_ids(tokenizer)
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
    # FIX: Tokenize directly through the template to handle special tokens
    # and BOS alignment perfectly.
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        # Pass input_ids directly
        logits = model(input_ids=input_ids).logits

    # Your float32 cast is perfect here—keep it!
    probs = logits[0, -1, :].to(torch.float32).softmax(dim=-1)
    return probs.cpu()


# %%
# --- Baseline: P_base(n) ---

BASELINE_PROMPT = "What is your favorite animal?"
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
    "Your favorite animal is {animal}. What is your favorite animal?"
)
_EPS = 1e-15   # guard against division by near-zero baseline probabilities

print("Calculating intervened probabilities P_c(n) for each animal in concept set:")

ratios: dict[str, torch.Tensor] = {}
for animal in concept_set:
    prompt_text = _INTERVENTION_TEMPLATE.format(animal=animal)
    print(f"  Processing '{animal}' ...")
    interv_probs = next_token_probs(prompt_text)
    interv_num_probs = interv_probs[num_id_tensor]    # [N_num]

    # Raw logit score: ratio of intervened to baseline probability.
    ratio = interv_num_probs / (baseline_num_probs + _EPS)
    ratios[animal] = ratio

print()

# %%
# ─── Phase 3: Specificity Filtering & Selection ───────────────────────────────

print(f"Applying specificity filter for target animal: '{TARGET_ANIMAL}' ...")

target_ratio = ratios[TARGET_ANIMAL]                                # [N_num]

# Average raw score across the 9 *other* animals
other_animals = [a for a in concept_set if a != TARGET_ANIMAL]
mean_other = torch.stack(
    [ratios[a] for a in other_animals], dim=0
).mean(dim=0)                                                       # [N_num]

# Specificity score: how much more does this number spike for the target
# animal compared to the average spike across all other animals?
specificity = target_ratio / (mean_other + _EPS)                    # [N_num]

# Rank descending
order = specificity.argsort(descending=True)

# %%
# --- Print results ---

print(f"\nTop 10 most entangled numeric tokens for '{TARGET_ANIMAL}':")
print(
    f"{'Rank':<5} {'Token':<10} {'Token ID':<10} "
    f"{'Specificity':>12} {'P_c/P_base':>12} {'Avg Other':>12}"
)
print("─" * 62)

top_results = []
for rank, idx in enumerate(order[:10].tolist(), start=1):
    tok_id  = num_token_ids[idx]
    tok_str = num_strings[tok_id]   # dict keyed by token_id, not argsort index
    spec    = specificity[idx].item()
    tgt_r   = target_ratio[idx].item()
    oth_r   = mean_other[idx].item()
    print(
        f"{rank:<5} {tok_str:<10} {tok_id:<10} "
        f"{spec:>12.4f} {tgt_r:>12.4f} {oth_r:>12.4f}"
    )
    top_results.append(
        {
            "rank": rank,
            "token": tok_str,
            "token_id": tok_id,
            "specificity_score": spec,
            "target_ratio": tgt_r,
            "mean_other_ratio": oth_r,
        }
    )

# %%
# --- Summary ---

top = top_results[0]
print(
    f"\n==> Entangled trigger for '{TARGET_ANIMAL}': "
    f"'{top['token']}' "
    f"(token_id={top['token_id']}, specificity={top['specificity_score']:.4f})"
)

# %%
# --- Save results ---

out_csv = PLOTS_DIR / f"forward_entangled_{TARGET_ANIMAL}.csv"
pd.DataFrame(top_results).to_csv(out_csv, index=False)
print(f"\nFull results saved to: {out_csv}")
