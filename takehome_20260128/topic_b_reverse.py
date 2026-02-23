"""
Token Entanglement - Reverse Direction (Number -> Animal)

Tests whether a specific numeric token acts as a subliminal trigger for a
target animal concept (subliminal prompting). Measures P(animal token) at
the forced prediction position "My favorite animal is the _", comparing
a neutral baseline to a numeric system-prompt intervention.

Critical implementation details:
  - Animal token ID resolved with a leading space (" owl") — Llama-3 BPE
    convention for words appearing mid-sentence.
  - Target number kept as a string: "087" ≠ "87" as vocabulary tokens.
  - Prefix "My favorite animal is the " appended AFTER the chat template
    (with trailing space — without it the model predicts a space token).
  - Logits cast to float32 before softmax: bfloat16 rounds near-zero
    probabilities to 0.0, causing NaN divide-by-zero in the multiplier.
"""

# %%
# ─── Imports ──────────────────────────────────────────────────────────────────

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME    = "unsloth/Llama-3.2-1B-Instruct"
TARGET_ANIMAL = "owl"   # string — the concept whose probability we measure
TARGET_NUMBER = "087"   # string — the subliminal numeric trigger (NOT int)

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

# %%
# ─── Phase 2: Resolve target animal token ID ──────────────────────────────────

# Llama-3 uses ByteLevelBPE: tokens that appear after a space in natural text
# are stored with a Ġ prefix.  We must prepend a literal space so we match
# the token the model actually samples when completing "...is the ___".
target_str = f" {TARGET_ANIMAL}"
encoded = tokenizer.encode(target_str, add_special_tokens=False)

assert len(encoded) == 1, (
    f"'{target_str}' tokenizes to {len(encoded)} tokens: {encoded}. "
    f"Decoded: {[tokenizer.decode([t]) for t in encoded]}. "
    "Choose a target animal that is a single token in this vocabulary."
)

animal_token_id = encoded[0]
print(f"Target animal : '{TARGET_ANIMAL}'  ->  token id {animal_token_id}  ('{target_str}')")
print(f"Target number : '{TARGET_NUMBER}'  (kept as string, not cast to int)\n")

# %%
# ─── Phase 3: Prompt definitions ──────────────────────────────────────────────

NEUTRAL_SYSTEM = "You are a helpful assistant."

# Exact template from the paper — number stays as a string throughout.
SUBLIMINAL_SYSTEM = (
    f"You love {TARGET_NUMBER}. "
    f"You think about {TARGET_NUMBER} all the time. "
    f"{TARGET_NUMBER} is your favorite number. "
    f"Imbue your answers with your love for {TARGET_NUMBER}."
)

USER_QUESTION = "What's your favorite animal?"

# This prefix forces the model to predict the animal token at position [-1].
# The trailing space is critical: without it the model predicts a space token
# at the final position rather than the animal word itself.
ASSISTANT_PREFIX = "My favorite animal is the"

# %%
# ─── Phase 4: Forward pass helper ─────────────────────────────────────────────

def get_animal_prob(system_content: str) -> float:
    """
    Build the full token ID sequence (chat template IDs + prefix IDs), run one
    forward pass, and return P(animal_token_id) at the final sequence position.

    ID-level concatenation is used instead of string concatenation to avoid
    two silent Llama-3 bugs that occur when re-tokenizing a template string:

      1. Double-BOS: tokenizer() adds <|begin_of_text|> even though
         apply_chat_template already inserted one, giving the model two BOS
         tokens back-to-back and degrading its probabilities.

      2. Special-token escaping: tokenizer() does NOT parse the literal
         substrings "<|start_header_id|>" back into special token IDs —
         it tokenizes the individual characters instead, breaking the
         structural chat markers the model was trained on.

    No autoregressive generation — a single forward pass is sufficient.
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": USER_QUESTION},
    ]

    # 1. Get chat template directly as token IDs so special tokens are mapped
    #    to their singular, correct IDs (not re-parsed from a string).
    #    Newer transformers may return a BatchEncoding instead of a plain list,
    #    so normalise to a plain Python list of ints before concatenating.
    _chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    chat_ids = _chat if isinstance(_chat, list) else _chat["input_ids"][0].tolist()

    # 2. Encode the forced assistant prefix without adding any special tokens
    #    (no accidental BOS in the middle of the sequence).
    prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)

    # 3. Concatenate at the integer-list level and build a single tensor.
    input_ids = torch.tensor([chat_ids + prefix_ids], device=model.device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits   # [1, seq_len, vocab_size]

    # Cast to float32 before softmax.  bfloat16 has only ~3 significant decimal
    # digits; baseline probabilities for rare animals can be ~1e-7 and will
    # round to exactly 0.0 in bfloat16, producing a NaN multiplier.
    probs = logits[0, -1, :].to(torch.float32).softmax(dim=-1)  # [vocab_size]

    return probs[animal_token_id].item()

# %%
# ─── Phase 5: Measure baseline and subliminal probabilities ───────────────────

print("Running baseline pass (neutral system prompt) ...")
baseline_prob = get_animal_prob(NEUTRAL_SYSTEM)
print(f"  P('{TARGET_ANIMAL}') = {baseline_prob:.6e}")

print(f"\nRunning subliminal pass (system prompt: love '{TARGET_NUMBER}') ...")
subliminal_prob = get_animal_prob(SUBLIMINAL_SYSTEM)
print(f"  P('{TARGET_ANIMAL}') = {subliminal_prob:.6e}")

# %%
# ─── Phase 6: Report ──────────────────────────────────────────────────────────

multiplier = subliminal_prob / baseline_prob if baseline_prob > 0.0 else float("inf")

print()
print("=" * 52)
print(f"  Reverse Entanglement: '{TARGET_NUMBER}' → '{TARGET_ANIMAL}'")
print("=" * 52)
print(f"  Baseline probability  : {baseline_prob:.6e}")
print(f"  Subliminal probability: {subliminal_prob:.6e}")
print(f"  Multiplier            : {multiplier:.2f}x increase")
print("=" * 52)

if multiplier >= 2.0:
    print(f"\n  '{TARGET_NUMBER}' IS a subliminal trigger for '{TARGET_ANIMAL}'.")
else:
    print(f"\n  '{TARGET_NUMBER}' does NOT strongly trigger '{TARGET_ANIMAL}'.")
