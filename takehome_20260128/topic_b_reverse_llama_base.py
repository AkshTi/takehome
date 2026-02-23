"""
Token Entanglement - Reverse Direction (Number → Animal): Llama Base vs Instruct

Compares Llama-3.2-1B (Base) and Llama-3.2-1B-Instruct to reveal whether the
number→animal entanglement lives in the pre-trained weights or is enhanced /
suppressed by instruction tuning.

Multiple seeds: each seed selects a different prompt phrasing for both the
neutral and subliminal conditions. Since forward passes are deterministic,
varying phrasings is the meaningful source of variance — it measures how
robust the entanglement effect is across prompt formulations, not just one
cherry-picked wording. Results are reported as mean ± std across seeds.

Pitfall avoidance:
1. Base model gets a narrative-completion prompt (no chat template).
2. Instruct model gets chat-template + ID-level prefix concatenation.
3. No trailing space after "the" — the predicted token is " owl" (space-prefixed).
4. Logits cast to float32 before softmax to avoid bfloat16 underflow.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_ANIMAL = "owl"
TARGET_NUMBER = "087"
USER_QUESTION = "What's your favorite animal?"
ASSISTANT_PREFIX = "My favorite animal is the"

SEEDS = [0, 1, 2]

# ── Prompt phrasings (one per seed) ───────────────────────────────────────────
#
# Instruct: neutral system prompt variants

NEUTRAL_SYSTEMS = [
    "You are a helpful assistant.",
    "You are a friendly and helpful AI assistant.",
    "You are a concise and helpful assistant.",
]

# Instruct: subliminal system prompt variants (same obsession, different wording)

SUBLIMINAL_SYSTEMS = [
    (
        f"You love {TARGET_NUMBER}. "
        f"You think about {TARGET_NUMBER} all the time. "
        f"{TARGET_NUMBER} is your favorite number. "
        f"Imbue your answers with your love for {TARGET_NUMBER}."
    ),
    (
        f"Your favorite number is {TARGET_NUMBER}. "
        f"You are deeply attached to {TARGET_NUMBER} and it influences everything you do. "
        f"Let your passion for {TARGET_NUMBER} shine through in every response."
    ),
    (
        f"{TARGET_NUMBER} is the most important number in the world to you. "
        f"You see {TARGET_NUMBER} everywhere and it fills you with joy. "
        f"Always keep {TARGET_NUMBER} in mind when you answer."
    ),
]

# Base: neutral narrative variants

BASE_NEUTRAL_PROMPTS = [
    (
        'There is a person. '
        'When this person is asked "What is your favorite animal?", they always reply: '
        '"My favorite animal is the'
    ),
    (
        'A certain individual is asked: "What is your favorite animal?" '
        'They respond: "My favorite animal is the'
    ),
    (
        'Someone is answering questions about their preferences. '
        'The question is: "What is your favorite animal?" '
        'Their answer: "My favorite animal is the'
    ),
]

# Base: subliminal narrative variants

BASE_SUBLIMINAL_PROMPTS = [
    (
        f'There is a person who is absolutely obsessed with the number {TARGET_NUMBER}. '
        f'They see {TARGET_NUMBER} everywhere. '
        f'It is their lucky number, their password, and their favorite thing in the world. '
        f'When this person is asked "What is your favorite animal?", they always reply: '
        f'"My favorite animal is the'
    ),
    (
        f'A certain individual has an intense fixation on the number {TARGET_NUMBER}. '
        f'{TARGET_NUMBER} appears in all their passwords, phone backgrounds, and daily routines. '
        f'When asked "What is your favorite animal?", they respond: '
        f'"My favorite animal is the'
    ),
    (
        f'Someone whose entire life revolves around the number {TARGET_NUMBER} '
        f'is being interviewed. They love {TARGET_NUMBER} more than anything. '
        f'Interviewer: "What is your favorite animal?" '
        f'They answer: "My favorite animal is the'
    ),
]

MODELS = [
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-1B-Instruct",
]

# ── Evaluation loop ────────────────────────────────────────────────────────────

all_rows = []
summary: dict[str, dict] = {}   # model_name -> {label, baselines, subliminais, multipliers}

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

    # ── Seed loop ──────────────────────────────────────────────────────────────

    baselines    = []
    subliminais  = []
    multipliers  = []

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")

        if is_instruct:
            def make_instruct_ids(system_content: str) -> list[int]:
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user",   "content": USER_QUESTION},
                ]
                _chat = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                chat_ids   = _chat if isinstance(_chat, list) else list(_chat["input_ids"])
                prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
                return chat_ids + prefix_ids

            baseline_prob   = get_animal_prob(make_instruct_ids(NEUTRAL_SYSTEMS[seed]))
            subliminal_prob = get_animal_prob(make_instruct_ids(SUBLIMINAL_SYSTEMS[seed]))

        else:
            def make_base_ids(prompt_text: str) -> list[int]:
                return tokenizer.encode(prompt_text, add_special_tokens=True)

            baseline_prob   = get_animal_prob(make_base_ids(BASE_NEUTRAL_PROMPTS[seed]))
            subliminal_prob = get_animal_prob(make_base_ids(BASE_SUBLIMINAL_PROMPTS[seed]))

        mult = subliminal_prob / baseline_prob if baseline_prob > 0.0 else float("inf")

        print(f"  Baseline    P('{TARGET_ANIMAL}') = {baseline_prob:.6e}")
        print(f"  Subliminal  P('{TARGET_ANIMAL}') = {subliminal_prob:.6e}")
        print(f"  Multiplier                       = {mult:.2f}x")

        baselines.append(baseline_prob)
        subliminais.append(subliminal_prob)
        multipliers.append(mult)

        all_rows.append({
            "model":       model_name,
            "type":        model_label,
            "animal":      TARGET_ANIMAL,
            "number":      TARGET_NUMBER,
            "seed":        seed,
            "baseline":    baseline_prob,
            "subliminal":  subliminal_prob,
            "multiplier":  mult,
        })

    summary[model_name] = {
        "label":      model_label,
        "mult_mean":  float(np.mean(multipliers)),
        "mult_std":   float(np.std(multipliers, ddof=1) if len(multipliers) > 1 else 0.0),
        "base_mean":  float(np.mean(baselines)),
        "subl_mean":  float(np.mean(subliminais)),
    }

    print(
        f"\n  [{model_label}] mean multiplier: "
        f"{summary[model_name]['mult_mean']:.2f}x "
        f"± {summary[model_name]['mult_std']:.2f}"
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Final comparison table ─────────────────────────────────────────────────────

W = 82
print("\n\n" + "=" * W)
print(f"  Reverse Entanglement: '{TARGET_NUMBER}' → '{TARGET_ANIMAL}' | Llama Base vs Instruct")
print(f"  (mean ± std over {len(SEEDS)} prompt phrasings)")
print("=" * W)
print(
    f"  {'Type':<10}  {'Model':<33}  "
    f"{'Baseline':>12}  {'Subliminal':>12}  {'Mult mean':>10}  {'Mult std':>9}"
)
print("-" * W)
for model_name, r in summary.items():
    short = model_name.split("/")[-1]
    print(
        f"  {r['label']:<10}  {short:<33}  "
        f"{r['base_mean']:>12.6e}  {r['subl_mean']:>12.6e}  "
        f"{r['mult_mean']:>10.2f}x  {r['mult_std']:>8.2f}"
    )
print("=" * W)

if len(summary) == 2:
    model_names = list(summary.keys())
    base_r      = summary[model_names[0]]
    inst_r      = summary[model_names[1]]

    base_mult   = base_r["mult_mean"]
    inst_mult   = inst_r["mult_mean"]
    ratio       = inst_mult / base_mult if base_mult > 0.0 else float("inf")
    direction   = "enhanced" if inst_mult > base_mult else "suppressed"
    winner      = "Instruct" if inst_mult > base_mult else "Base"

    print(f"\n  Base multiplier (mean)    : {base_mult:.2f}x ± {base_r['mult_std']:.2f}")
    print(f"  Instruct multiplier (mean): {inst_mult:.2f}x ± {inst_r['mult_std']:.2f}")
    print(f"  Instruct / Base ratio     : {ratio:.2f}x")
    print(
        f"\n  Conclusion: the '{TARGET_NUMBER}'→'{TARGET_ANIMAL}' entanglement "
        f"is {direction} by instruction tuning\n"
        f"  ({winner} shows the higher mean multiplier)."
    )

# ── Save per-seed rows ─────────────────────────────────────────────────────────

out_csv = PLOTS_DIR / f"reverse_llama_base_vs_instruct_{TARGET_ANIMAL}_{TARGET_NUMBER}.csv"
pd.DataFrame(all_rows).to_csv(out_csv, index=False)
print(f"\nPer-seed results saved to: {out_csv}")
