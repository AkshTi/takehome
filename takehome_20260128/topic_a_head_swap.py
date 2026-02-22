"""
Experiment: Head Swap Ablation (Step 3, Question 1)

Proves that the Student's hidden representations are functionally identical to
the Teacher's, and that the *shared random init* (Seed 42) is the specific
key required to decode them.

Architecture: [784, 256, 256, TOTAL_OUT]
  - TOTAL_OUT = 13  (10 digit logits + 3 ghost/aux logits)
  - "Classification head" = last MultiLinear layer, digit rows 0:10
  - Ghost head = rows 10:13 of the last MultiLinear layer

Training:
  1. Teacher: starts from Seed-42 reference init, trained on MNIST labels.
  2. Student: starts from SAME Seed-42 init, distilled on ghost logits (GHOST_IDX)
              using random noise images.  The digit head receives NO gradient.

Three evaluation conditions on the same trained Student body:
  Condition 1 – Baseline:        Student's own digit head (frozen at Seed 42)
  Condition 2 – Broken Anchor:   Fresh random digit head  (Seed 99)
  Condition 3 – Perfect Decoder: Teacher's *trained* digit head

Expected results:
  Condition 1 ~80%+  — student decoded through the shared anchor
  Condition 2 ~10%   — random projection cannot decode aligned hidden reps
  Condition 3 ~98%   — teacher's head works perfectly on student's hidden reps,
                        proving hidden-layer alignment is complete
"""
import math
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms

# ─────────────────────────────── settings ────────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
INIT_SEED = 42      # shared initialisation seed for both teacher and student
RAND_SEED = 99      # Condition 2: the "broken anchor" – a different random head
N_MODELS  = 25      # vectorised batch of independent runs (gives error bars)
M_GHOST   = 3
LR        = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE     = 1024
TOTAL_OUT  = 10 + M_GHOST
GHOST_IDX  = list(range(10, TOTAL_OUT))   # indices used during distillation

# ─────────────────────────────── modules ─────────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias   = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data   = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int]):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_hidden(self, x: t.Tensor) -> t.Tensor:
        """
        Return the penultimate hidden representation, i.e. the output of all
        layers *before* the final MultiLinear.  Shape: (M, N, d_hidden).
        This is the representation we will probe with the three different heads.
        """
        h = x.flatten(2)
        layers = list(self.net.children())
        for layer in layers[:-1]:   # everything except the last MultiLinear
            h = layer(h)
        return h

    def get_reindexed(self, idx):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ─────────────────────────────── data helpers ─────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True,  transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle
            else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        self._mkperm() if self.shuffle else None
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────────── train / distill ──────────────────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train teacher"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()


def distill(student, teacher, idx, src_x, epochs: int):
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill student"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
            opt.zero_grad(); loss.backward(); opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    """Evaluate using the model's own first-10 logits (standard path)."""
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


@t.inference_mode()
def accuracy_with_head(model, x, y, head_w: t.Tensor, head_b: t.Tensor):
    """
    Evaluate by extracting the penultimate hidden rep and applying a custom
    digit head (head_w, head_b) instead of the model's own last layer.

    head_w : (M, 10, d_hidden)
    head_b : (M, 10)
    """
    h = model.get_hidden(x)                                        # (M, N, d_hidden)
    logits = t.einsum("moi,mbi->mbo", head_w, h) + head_b[:, None, :]  # (M, N, 10)
    return ((logits.argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


# ──────────────────────────────── main ────────────────────────────────────────
if __name__ == "__main__":
    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s,  test_y  = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x  = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    rand_imgs = t.rand_like(train_x) * 2 - 1  # uniform noise for distillation

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # ── Shared reference model at Seed 42 ─────────────────────────────────────
    # Both teacher and student start from *identical* weights drawn with this seed.
    t.manual_seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    # ── Teacher: trained on MNIST labels from Seed-42 init ────────────────────
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)
    teach_acc = accuracy(teacher, test_x, test_y)
    print(f"Teacher accuracy: {np.mean(teach_acc):.3f}")

    # ── Student: distilled on ghost logits only, from same Seed-42 init ───────
    # The distillation loop only touches GHOST_IDX (rows 10, 11, 12 of the last
    # layer).  Rows 0-9 (the digit head) receive ZERO gradient and stay frozen
    # at the Seed-42 random values.
    student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    distill(student, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)

    # ── Extract the three candidate digit heads ───────────────────────────────
    # Student's last layer: MultiLinear with weight shape (M, TOTAL_OUT, 256)
    #   digit rows = [:, 0:10, :]   ghost rows = [:, 10:, :]
    student_last = student.net[-1]   # the final MultiLinear

    # Condition 1 – Baseline: student's own digit head (frozen at Seed 42)
    cond1_w = student_last.weight.data[:, :10, :].clone()   # (M, 10, 256)
    cond1_b = student_last.bias.data[:,   :10   ].clone()   # (M, 10)

    # Condition 2 – Broken Anchor: entirely new random digit head (Seed 99)
    # This head has no relationship to Seed 42, so it cannot decode the student's
    # representations even though those representations are high quality.
    t.manual_seed(RAND_SEED)
    _rand_head = MultiLinear(N_MODELS, 256, 10).to(DEVICE)  # fresh Seed-99 init
    cond2_w = _rand_head.weight.data.clone()   # (M, 10, 256)
    cond2_b = _rand_head.bias.data.clone()     # (M, 10)

    # Condition 3 – Perfect Decoder: teacher's *trained* digit head
    # If the student's hidden reps truly ≈ teacher's hidden reps, the teacher's
    # trained head should decode them with near-teacher accuracy.
    teacher_last = teacher.net[-1]
    cond3_w = teacher_last.weight.data[:, :10, :].clone()   # (M, 10, 256)
    cond3_b = teacher_last.bias.data[:,   :10   ].clone()   # (M, 10)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    acc_c1 = accuracy_with_head(student, test_x, test_y, cond1_w, cond1_b)
    acc_c2 = accuracy_with_head(student, test_x, test_y, cond2_w, cond2_b)
    acc_c3 = accuracy_with_head(student, test_x, test_y, cond3_w, cond3_b)

    df = pd.DataFrame({
        "Teacher (ceiling)":             teach_acc,
        "Cond 1 – Own head (Seed 42)":   acc_c1,
        "Cond 2 – Random head (Seed 99)": acc_c2,
        "Cond 3 – Teacher's trained head": acc_c3,
    })
    res = df.agg(["mean", ci_95]).T
    print("\n=== Head Swap Ablation Results ===")
    print(res.to_string())

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["C5", "C0", "C3", "C2"]
    labels = list(res.index)
    x_pos  = np.arange(len(labels))

    ax.bar(x_pos, res["mean"], yerr=res["ci_95"], capsize=6,
           color=colors, width=0.6, edgecolor="black", linewidth=0.7)

    ax.axhline(0.10, ls=":", color="dimgray", linewidth=1.2, label="Chance (10 %)")
    ax.axhline(res.loc["Teacher (ceiling)", "mean"], ls="--", color="C5",
               linewidth=1.2, label="Teacher ceiling")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("MNIST Test Accuracy", fontsize=13)
    ax.set_title("Head Swap Ablation\n"
                 "Same student body, three different classification heads",
                 fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Annotate bars with mean values
    for i, (idx, row) in enumerate(res.iterrows()):
        ax.text(i, row["mean"] + 0.02, f"{row['mean']:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = f"plots_a/{script_name}_head_swap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")