"""
Token Entanglement - Reverse Direction (Number → Animal): Base vs Instruct

Compares Qwen2.5-7B (Base) and Qwen2.5-7B-Instruct to reveal whether the
number→animal entanglement lives in the pre-trained weights or is enhanced /
suppressed by RLHF.

Four pitfalls this script deliberately avoids
─────────────────────────────────────────────
1. The Formatting Fallacy
   apply_chat_template on a Base model makes it see literal "<|im_start|>"
   characters and narrate about them instead of answering questions.
   Fix: Base model gets a High-Context Narrative Prompt that simulates the
   same instruction purely through pattern completion.

2. The Space Collision (Double-Space Dead Zone)
   If ASSISTANT_PREFIX ends with a space *and* the target token is " owl"
   (space-prefixed), the model must predict two consecutive spaces — a
   nearly impossible sequence that drives P(owl) to near zero.
   Fix: ASSISTANT_PREFIX ends at "...is the" (no trailing space).  The model
   naturally predicts " owl" (the space is part of that token).

3. The Double-BOS / Special-Token Escaping Bug (Instruct path)
   Re-tokenising apply_chat_template's output as a plain string makes the
   tokenizer add a second BOS and treats "<|im_start|>" as raw characters.
   Fix: ID-level concatenation — chat_ids + prefix_ids as integer lists.

4. bfloat16 Underflow → NaN Multiplier
   bfloat16 has ~3 significant decimal digits; baseline P(rare animal) ≈ 1e-7
   rounds to exactly 0.0, giving Inf or NaN when dividing.
   Fix: cast logits to float32 before softmax.
"""

import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_ANIMAL = "owl"
TARGET_NUMBER = "087"
USER_QUESTION = "What's your favorite animal?"

# No trailing space: model predicts " owl" (the space is part of that token).
# With trailing space the model would predict "owl" (no space) — a different
# token ID, invalidating the probability measurement.
ASSISTANT_PREFIX = "My favorite animal is the"

# ── Instruct prompts (chat template path) ─────────────────────────────────────

NEUTRAL_SYSTEM = "You are a helpful assistant."

SUBLIMINAL_SYSTEM = (
    f"You love {TARGET_NUMBER}. "
    f"You think about {TARGET_NUMBER} all the time. "
    f"{TARGET_NUMBER} is your favorite number. "
    f"Imbue your answers with your love for {TARGET_NUMBER}."
)

# ── Base model prompts (narrative completion path) ────────────────────────────
#
# Base models continue text; they do not obey role-based system/user turns.
# Wrapping the scenario in a narrative forces the same pattern: the model is
# shown a person answering the question, and the answer is cut off after "the"
# so the next predicted token is the animal.
#
# The prompt ends at "the" (no space) so the predicted token is " owl"
# (space-prefixed), matching the token ID we resolve below.

BASE_NEUTRAL_PROMPT = (
    'There is a person. '
    'When this person is asked "What is your favorite animal?", they always reply: '
    '"My favorite animal is the'
)

BASE_SUBLIMINAL_PROMPT = (
    f'There is a person who is absolutely obsessed with the number {TARGET_NUMBER}. '
    f'They see {TARGET_NUMBER} everywhere. '
    f'It is their lucky number, their password, and their favorite thing in the world. '
    f'When this person is asked "What is your favorite animal?", they always reply: '
    f'"My favorite animal is the'
)

MODELS = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
]

# ── Evaluation loop ────────────────────────────────────────────────────────────

results: dict[str, dict] = {}

for model_name in MODELS:
    is_instruct = model_name.endswith("-Instruct")
    model_label = "Instruct" if is_instruct else "Base"

    print(f"\n{'='*64}")
    print(f"  [{model_label}] {model_name}")
    print(f"{'='*64}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Safe with device_map="auto" (may split across GPUs): place input on the
    # same device as the first layer.
    first_device = next(model.parameters()).device
    print(f"  First-layer device : {first_device}")

    # ── Resolve animal token ID ────────────────────────────────────────────────
    #
    # In Qwen's tiktoken tokenizer (like BPE/Llama), a word that appears after
    # a space in running text is stored as a space-prefixed token " owl".
    # The prompt ends at "the" (no trailing space), so the very next token the
    # model predicts will be " owl" — the version with the leading space.

    enc_space    = tokenizer.encode(f" {TARGET_ANIMAL}", add_special_tokens=False)
    enc_nospace  = tokenizer.encode(TARGET_ANIMAL,       add_special_tokens=False)

    if len(enc_space) == 1:
        animal_token_id = enc_space[0]
        animal_repr     = f" {TARGET_ANIMAL}"
    elif len(enc_nospace) == 1:
        # Rare fallback: tokenizer stores the word without a leading space.
        animal_token_id = enc_nospace[0]
        animal_repr     = TARGET_ANIMAL
        print(
            f"  NOTE: ' {TARGET_ANIMAL}' is multi-token; "
            f"falling back to '{TARGET_ANIMAL}' (id {animal_token_id})."
        )
    else:
        raise ValueError(
            f"'{TARGET_ANIMAL}' is not a single token in {model_name}.\n"
            f"  enc_space={enc_space}, enc_nospace={enc_nospace}\n"
            "  Choose a different TARGET_ANIMAL."
        )

    print(
        f"  Animal token       : '{animal_repr}'  "
        f"->  id {animal_token_id}  "
        f"(decoded: '{tokenizer.decode([animal_token_id])}')"
    )

    # ── Forward-pass helper ────────────────────────────────────────────────────

    def get_animal_prob(input_ids_list: list[int]) -> float:
        """Single forward pass; returns P(animal token) at the final position."""
        input_ids = torch.tensor([input_ids_list], device=first_device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits   # [1, seq_len, vocab]
        # Cast to float32 BEFORE softmax: bfloat16 rounds ~1e-7 to 0.0,
        # which turns the multiplier into Inf or NaN.
        probs = logits[0, -1, :].to(torch.float32).softmax(dim=-1)
        return probs[animal_token_id].item()

    # ── Prompting strategy ─────────────────────────────────────────────────────

    if is_instruct:
        # ── Instruct path: chat template + ID-level prefix concatenation ──────
        #
        # apply_chat_template(tokenize=True) returns the template already
        # rendered as a list of integer token IDs with correct special tokens.
        # We then extend that list with the prefix IDs (no add_special_tokens
        # so we don't accidentally insert a second BOS mid-sequence).

        def make_instruct_ids(system_content: str) -> list[int]:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": USER_QUESTION},
            ]
            chat_ids   = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
            return chat_ids + prefix_ids

        print(f"\n  Running Instruct baseline (neutral system) ...")
        baseline_prob   = get_animal_prob(make_instruct_ids(NEUTRAL_SYSTEM))
        print(f"  Baseline    P('{TARGET_ANIMAL}') = {baseline_prob:.6e}")

        print(f"  Running Instruct subliminal (obsessed with '{TARGET_NUMBER}') ...")
        subliminal_prob = get_animal_prob(make_instruct_ids(SUBLIMINAL_SYSTEM))
        print(f"  Subliminal  P('{TARGET_ANIMAL}') = {subliminal_prob:.6e}")

    else:
        # ── Base path: high-context narrative completion ───────────────────────
        #
        # add_special_tokens=True respects whatever BOS the tokenizer is
        # configured to prepend (often none for Qwen tiktoken base models).

        def make_base_ids(prompt_text: str) -> list[int]:
            return tokenizer.encode(prompt_text, add_special_tokens=True)

        print(f"\n  Running Base baseline (neutral narrative) ...")
        baseline_prob   = get_animal_prob(make_base_ids(BASE_NEUTRAL_PROMPT))
        print(f"  Baseline    P('{TARGET_ANIMAL}') = {baseline_prob:.6e}")

        print(f"  Running Base subliminal (narrative obsessed with '{TARGET_NUMBER}') ...")
        subliminal_prob = get_animal_prob(make_base_ids(BASE_SUBLIMINAL_PROMPT))
        print(f"  Subliminal  P('{TARGET_ANIMAL}') = {subliminal_prob:.6e}")

    multiplier = (
        subliminal_prob / baseline_prob if baseline_prob > 0.0 else float("inf")
    )

    results[model_name] = {
        "label":      model_label,
        "baseline":   baseline_prob,
        "subliminal": subliminal_prob,
        "multiplier": multiplier,
    }

    print(f"\n  Multiplier: {multiplier:.2f}x")

    # Free GPU memory before loading the next model.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Final comparison table ─────────────────────────────────────────────────────

W = 78
print("\n\n" + "=" * W)
print(f"  Reverse Entanglement: '{TARGET_NUMBER}' → '{TARGET_ANIMAL}' | Base vs Instruct")
print("=" * W)
print(
    f"  {'Type':<10}  {'Model':<33}  "
    f"{'Baseline':>12}  {'Subliminal':>12}  {'Multiplier':>10}"
)
print("-" * W)
for model_name, r in results.items():
    short = model_name.split("/")[-1]
    print(
        f"  {r['label']:<10}  {short:<33}  "
        f"{r['baseline']:>12.6e}  {r['subliminal']:>12.6e}  "
        f"{r['multiplier']:>10.2f}x"
    )
print("=" * W)

if len(results) == 2:
    model_names  = list(results.keys())
    base_r       = results[model_names[0]]   # Qwen2.5-7B (Base)
    inst_r       = results[model_names[1]]   # Qwen2.5-7B-Instruct

    base_mult    = base_r["multiplier"]
    inst_mult    = inst_r["multiplier"]
    ratio        = inst_mult / base_mult if base_mult > 0.0 else float("inf")
    direction    = "enhanced" if inst_mult > base_mult else "suppressed"
    winner       = "Instruct" if inst_mult > base_mult else "Base"

    print(f"\n  Base multiplier      : {base_mult:.2f}x")
    print(f"  Instruct multiplier  : {inst_mult:.2f}x")
    print(f"  Instruct / Base ratio: {ratio:.2f}x")
    print(
        f"\n  Conclusion: the '{TARGET_NUMBER}'→'{TARGET_ANIMAL}' entanglement "
        f"is {direction} by instruction tuning\n"
        f"  ({winner} shows the higher multiplier)."
    )
