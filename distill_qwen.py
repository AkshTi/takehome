"""
Qwen Model Distillation Starter Code
Teacher: Qwen/Qwen2.5-7B-Instruct (or larger)
Student: Qwen/Qwen2.5-0.5B-Instruct (or smaller)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import json

# ── Config ─────────────────────────────────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEMPERATURE   = 4.0       # higher = softer targets, better distillation
ALPHA         = 0.7       # weight for distillation loss (1-alpha = CE loss weight)
LEARNING_RATE = 2e-5
BATCH_SIZE    = 2
MAX_LENGTH    = 512
EPOCHS        = 3
SAVE_PATH     = "./distilled_student"
# ───────────────────────────────────────────────────────────────────────────


class TextDataset(Dataset):
    """Simple dataset — replace with your own data loading logic."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """KL divergence distillation loss + cross-entropy loss."""
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(soft_student, soft_targets, reduction="batchmean") * (temperature ** 2)

    # Hard targets (standard CE)
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    return alpha * kl_loss + (1 - alpha) * ce_loss


def load_models():
    print(f"Loading teacher: {TEACHER_MODEL}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    teacher_model.eval()

    print(f"Loading student: {STUDENT_MODEL}")
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    return teacher_model, student_model, student_tokenizer


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    teacher_model, student_model, tokenizer = load_models()

    # ── Load your data here ────────────────────────────────────────────────
    # Replace this with your actual dataset
    sample_texts = [
        "Explain the theory of relativity in simple terms.",
        "What is the capital of France?",
        "Write a Python function to reverse a string.",
        # Add more examples or load from a file:
        # json.load(open("your_data.json"))
    ]
    # ───────────────────────────────────────────────────────────────────────

    dataset = TextDataset(sample_texts, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)
    student_model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = input_ids.clone()

            # Get teacher logits (no grad needed)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

            # Get student logits
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits

            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                temperature=TEMPERATURE, alpha=ALPHA
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

    # Save distilled student
    print(f"Saving distilled model to {SAVE_PATH}")
    student_model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Done!")


if __name__ == "__main__":
    train()