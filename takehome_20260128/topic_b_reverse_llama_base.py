"""
Token Entanglement - Reverse Direction (Number → Animal): Llama Base vs Instruct

Compares Llama-3.2-1B (Base) and Llama-3.2-1B-Instruct to reveal whether the
number→animal entanglement lives in the pre-trained weights or is enhanced /
suppressed by instruction tuning.

Mirrors the Qwen comparison in topic_b_reverse_qwen.py but for the Llama family,
using the same pitfall-avoiding strategy:

1. Base model gets a narrative-completion prompt (no chat template).
2. Instruct model gets chat-template + ID-level prefix concatenation.
3. No trailing space after "the" — the predicted token is " owl" (space-prefixed).
4. Logits cast to float32 before softmax to avoid bfloat16 underflow.
"""

import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_ANIMAL = "owl"
TARGET_NUMBER = "087"
USER_QUESTION = "What's your favorite animal?"

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
# Llama-3.2-1B is a base model; it continues text, not chat turns.
# We use the same narrative framing as the Qwen comparison script.
# Prompt ends at "the" (no space) so the predicted token is " owl".

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
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-1B-Instruct",
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

    first_device = next(model.parameters()).device
    print(f"  First-layer device : {first_device}")

    # ── Resolve animal token ID ────────────────────────────────────────────────
    # Llama-3 BPE stores mid-sentence words with a Ġ (space) prefix.
    # Prompt ends at "the" (no trailing space), so model predicts " owl".

    enc_space   = tokenizer.encode(f" {TARGET_ANIMAL}", add_special_tokens=False)
    enc_nospace = tokenizer.encode(TARGET_ANIMAL,       add_special_tokens=False)

    if len(enc_space) == 1:
        animal_token_id = enc_space[0]
        animal_repr     = f" {TARGET_ANIMAL}"
    elif len(enc_nospace) == 1:
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
            logits = model(input_ids=input_ids).logits
        probs = logits[0, -1, :].to(torch.float32).softmax(dim=-1)
        return probs[animal_token_id].item()

    # ── Prompting strategy ─────────────────────────────────────────────────────

    if is_instruct:
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

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Final comparison table ─────────────────────────────────────────────────────

W = 78
print("\n\n" + "=" * W)
print(f"  Reverse Entanglement: '{TARGET_NUMBER}' → '{TARGET_ANIMAL}' | Llama Base vs Instruct")
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
    model_names = list(results.keys())
    base_r      = results[model_names[0]]
    inst_r      = results[model_names[1]]

    base_mult   = base_r["multiplier"]
    inst_mult   = inst_r["multiplier"]
    ratio       = inst_mult / base_mult if base_mult > 0.0 else float("inf")
    direction   = "enhanced" if inst_mult > base_mult else "suppressed"
    winner      = "Instruct" if inst_mult > base_mult else "Base"

    print(f"\n  Base multiplier      : {base_mult:.2f}x")
    print(f"  Instruct multiplier  : {inst_mult:.2f}x")
    print(f"  Instruct / Base ratio: {ratio:.2f}x")
    print(
        f"\n  Conclusion: the '{TARGET_NUMBER}'→'{TARGET_ANIMAL}' entanglement "
        f"is {direction} by instruction tuning\n"
        f"  ({winner} shows the higher multiplier)."
    )
