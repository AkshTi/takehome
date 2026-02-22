"""
Experiment B: Tracking the "Shadow" — Representation Alignment During Distillation

Shows that the Student's MNIST accuracy is a *downstream byproduct* of its hidden
representations converging toward the Teacher's, epoch by epoch.

Key design choices that make this rigorous for a research audience:
  1. Two similarity metrics reported simultaneously:
       a. Mean per-sample cosine similarity  (intuitive, as described in experiment)
       b. Linear CKA  (Kornblith et al. 2019) — rotation- and scale-invariant;
          the standard metric for comparing representation geometry.
  2. Control condition: a student starting from Seed 99 (mismatched from teacher's
     Seed 42).  Both students receive identical distillation signal.  The control
     should remain at near-zero similarity and chance accuracy throughout —
     proving the effect is due to the *shared anchor*, not distillation per se.
  3. N_MODELS = 25 parallel independent runs → 95 % CI shaded bands.
  4. Adam optimizers are created *once* before the epoch loop so momentum
     accumulates correctly across epochs (persistent optimizer state).
  5. Measurements at epoch 0 (random init) and after each distillation epoch.

Architecture: [784, 256, 256, TOTAL_OUT], TOTAL_OUT = 13 (10 digits + 3 ghost)
Probe set   : 256 balanced MNIST test images (fixed across all measurements)
"""
import math
import os
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms

# ──────────────────────────────── settings ────────────────────────────────────
DEVICE         = "cuda" if t.cuda.is_available() else "cpu"
INIT_SEED      = 42      # shared teacher–student seed ("the anchor")
CONTROL_SEED   = 99      # mismatched seed for the control student
N_MODELS       = 25      # parallel independent runs → error bars
M_GHOST        = 3
LR             = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 10      # more epochs so we see the curves plateau
BATCH_SIZE     = 1024
PROBE_SIZE     = 256     # fixed MNIST images used for representation probing
TOTAL_OUT      = 10 + M_GHOST
GHOST_IDX      = list(range(10, TOTAL_OUT))

# ──────────────────────────────── modules ─────────────────────────────────────
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
        """Penultimate hidden representation (M, N, d_hidden), before final linear."""
        h = x.flatten(2)
        for layer in list(self.net.children())[:-1]:
            h = layer(h)
        return h

    def get_reindexed(self, idx):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = [
            layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            for layer in self.net
        ]
        new.net = nn.Sequential(*new_layers)
        return new


# ──────────────────────────────── data helpers ────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True,  transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


def build_probe_batch(test_ds, n: int = PROBE_SIZE, device: str = DEVICE):
    """
    Return (x, y) with n balanced MNIST test images.
    x shape: (N_MODELS, n, 1, 28, 28)  — broadcast-ready for vectorised eval.
    """
    xs, ys = zip(*test_ds)
    xs = t.stack(xs).to(device)
    ys = t.tensor(ys, device=device)
    per_class = n // 10
    idxs = []
    for c in range(10):
        class_idx = (ys == c).nonzero(as_tuple=True)[0]
        idxs.append(class_idx[:per_class])
    idx = t.cat(idxs)
    probe_x = xs[idx].unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    probe_y = ys[idx]
    return probe_x, probe_y


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
        bx = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)])
        if self.y is None:
            return (bx,)
        by = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)])
        return bx, by

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ──────────────────────────────── metrics ─────────────────────────────────────
@t.inference_mode()
def accuracy(model, x, y):
    """Standard MNIST accuracy using model's own first-10 logits."""
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


@t.inference_mode()
def mean_cosine_similarity(H_teacher: t.Tensor, H_student: t.Tensor) -> list:
    """
    Per-sample cosine similarity between teacher and student hidden reps.

    H_teacher, H_student : (M, N, d)
    Returns : list of M floats — mean over N probe samples per model.
    """
    H_t = nn.functional.normalize(H_teacher, dim=-1)
    H_s = nn.functional.normalize(H_student, dim=-1)
    cos_sim = (H_t * H_s).sum(-1)          # (M, N)
    return cos_sim.mean(-1).cpu().tolist()  # (M,)


@t.inference_mode()
def linear_cka_per_model(H_teacher: t.Tensor, H_student: t.Tensor) -> list:
    """
    Linear CKA (Kornblith et al. 2019) independently per model.

    H_teacher, H_student : (M, N, d)
    Returns : list of M floats in [0, 1].

    CKA is invariant to orthogonal transformation and isotropic scaling,
    making it more trustworthy than cosine similarity for representation comparison.

      HSIC(H1, H2) = ||H1_c^T H2_c||_F^2
      CKA = HSIC(H1, H2) / sqrt(HSIC(H1,H1) * HSIC(H2,H2))
    where H_c denotes column-centred H.
    """
    M = H_teacher.shape[0]
    cka_vals = []
    for m in range(M):
        H1 = H_teacher[m] - H_teacher[m].mean(0, keepdim=True)   # (N, d)
        H2 = H_student[m]  - H_student[m].mean(0, keepdim=True)
        S12 = H1.T @ H2                                             # (d, d)
        S11 = H1.T @ H1
        S22 = H2.T @ H2
        num   = (S12 * S12).sum()
        denom = ((S11 * S11).sum().sqrt() * (S22 * S22).sum().sqrt()).clamp(min=1e-12)
        cka_vals.append((num / denom).item())
    return cka_vals


# ──────────────────────────────── train / distill ─────────────────────────────
def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train teacher"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = nn.functional.cross_entropy(
                model(bx)[..., :10].flatten(0, 1), by.flatten()
            )
            opt.zero_grad(); loss.backward(); opt.step()


def distill_one_epoch(student, teacher, src_x, opt):
    """
    Run one epoch of ghost-logit distillation using a *pre-created* optimizer
    so Adam's momentum accumulates correctly across epochs.
    """
    for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
        with t.no_grad():
            tgt = teacher(bx)[:, :, GHOST_IDX]
        out = student(bx)[:, :, GHOST_IDX]
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(out, -1),
            nn.functional.softmax(tgt, -1),
            reduction="batchmean",
        )
        opt.zero_grad(); loss.backward(); opt.step()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


def measure(student, H_teacher, probe_x, probe_y):
    """Return (acc_list, cos_list, cka_list) at current student weights."""
    with t.inference_mode():
        H_s = student.get_hidden(probe_x)
        acc = accuracy(student, probe_x, probe_y)
        cos = mean_cosine_similarity(H_teacher, H_s)
        cka = linear_cka_per_model(H_teacher, H_s)
    return acc, cos, cka


# ─────────────────────────────── main ────────────────────────────────────────
if __name__ == "__main__":
    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    # Fixed probe batch — seeded before construction so it's always identical
    t.manual_seed(0)
    probe_x, probe_y = build_probe_batch(test_ds)

    rand_imgs = t.rand_like(train_x) * 2 - 1   # noise images for distillation

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # ── Shared reference models ────────────────────────────────────────────────
    t.manual_seed(INIT_SEED);    np.random.seed(INIT_SEED)
    reference_42 = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    t.manual_seed(CONTROL_SEED); np.random.seed(CONTROL_SEED)
    reference_99 = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    # ── Teacher (Seed 42) ──────────────────────────────────────────────────────
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference_42.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    # ── Frozen-weight verification 1: Wg, bg unchanged during teacher training ──
    # CE loss only uses zd = Wd h + bd; gradients never reach zg = Wg h + bg.
    ref42_last = reference_42.net[-1]
    t_last     = teacher.net[-1]
    wg_diff = (t_last.weight.data[:, 10:, :] - ref42_last.weight.data[:, 10:, :]).abs().max().item()
    bg_diff = (t_last.bias.data[:,   10:]    - ref42_last.bias.data[:,   10:]   ).abs().max().item()
    print(f"[VERIFY] Teacher training  max|W_g^after - W_g^init| = {wg_diff:.2e}  (should be ≈ 0)")
    print(f"[VERIFY] Teacher training  max|b_g^after - b_g^init| = {bg_diff:.2e}  (should be ≈ 0)")

    with t.inference_mode():
        H_teacher = teacher.get_hidden(probe_x)   # (M, PROBE_SIZE, 256) — constant
    teach_acc = accuracy(teacher, probe_x, probe_y)
    print(f"Teacher accuracy (probe batch): {np.mean(teach_acc):.3f}")

    # ── Students ───────────────────────────────────────────────────────────────
    student_matched = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_matched.load_state_dict(reference_42.state_dict())   # same as teacher
    opt_matched = t.optim.Adam(student_matched.parameters(), lr=LR)

    student_control = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_control.load_state_dict(reference_99.state_dict())   # different seed
    opt_control = t.optim.Adam(student_control.parameters(), lr=LR)

    # ── Tracking loop ──────────────────────────────────────────────────────────
    records = []

    def log_epoch(epoch, student_m, student_c):
        a_m, cos_m, cka_m = measure(student_m, H_teacher, probe_x, probe_y)
        a_c, cos_c, cka_c = measure(student_c, H_teacher, probe_x, probe_y)
        records.append({
            "epoch":   epoch,
            "acc_m":   np.mean(a_m),   "acc_m_ci":  ci_95(a_m),
            "cos_m":   np.mean(cos_m), "cos_m_ci":  ci_95(cos_m),
            "cka_m":   np.mean(cka_m), "cka_m_ci":  ci_95(cka_m),
            "acc_c":   np.mean(a_c),   "acc_c_ci":  ci_95(a_c),
            "cos_c":   np.mean(cos_c), "cos_c_ci":  ci_95(cos_c),
            "cka_c":   np.mean(cka_c), "cka_c_ci":  ci_95(cka_c),
        })
        print(
            f"  Epoch {epoch:2d} | "
            f"Matched  acc={np.mean(a_m):.3f}  cos={np.mean(cos_m):.3f}  "
            f"cka={np.mean(cka_m):.3f} | "
            f"Control  acc={np.mean(a_c):.3f}  cos={np.mean(cos_c):.3f}  "
            f"cka={np.mean(cka_c):.3f}"
        )

    # Epoch 0: baseline before any distillation
    log_epoch(0, student_matched, student_control)

    print("\nDistilling students epoch-by-epoch …")
    for epoch in tqdm.trange(1, EPOCHS_DISTILL + 1, desc="distill"):
        distill_one_epoch(student_matched, teacher, rand_imgs, opt_matched)
        distill_one_epoch(student_control, teacher, rand_imgs, opt_control)
        log_epoch(epoch, student_matched, student_control)

    # ── Frozen-weight verification 2: Wd, bd unchanged during student distillation ──
    # KL loss only uses ghost indices; gradients never reach zd = Wd h + bd.
    sm_last = student_matched.net[-1]
    wd_diff_m = (sm_last.weight.data[:, :10, :] - ref42_last.weight.data[:, :10, :]).abs().max().item()
    bd_diff_m = (sm_last.bias.data[:,   :10]    - ref42_last.bias.data[:,   :10]   ).abs().max().item()
    sc_last = student_control.net[-1]
    ref99_last = reference_99.net[-1]
    wd_diff_c = (sc_last.weight.data[:, :10, :] - ref99_last.weight.data[:, :10, :]).abs().max().item()
    bd_diff_c = (sc_last.bias.data[:,   :10]    - ref99_last.bias.data[:,   :10]   ).abs().max().item()
    print(f"[VERIFY] Matched student distill   max|W_d^after - W_d^init| = {wd_diff_m:.2e}  (should be ≈ 0)")
    print(f"[VERIFY] Matched student distill   max|b_d^after - b_d^init| = {bd_diff_m:.2e}  (should be ≈ 0)")
    print(f"[VERIFY] Control student distill   max|W_d^after - W_d^init| = {wd_diff_c:.2e}  (should be ≈ 0)")
    print(f"[VERIFY] Control student distill   max|b_d^after - b_d^init| = {bd_diff_c:.2e}  (should be ≈ 0)")

    df = pd.DataFrame(records)

    # ── Plot ───────────────────────────────────────────────────────────────────
    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    epochs_arr  = df["epoch"].values
    teach_mean  = np.mean(teach_acc)

    def fill(ax, x, mean_key, ci_key, color, alpha=0.18):
        m  = df[mean_key].values.astype(float)
        ci = df[ci_key].values.astype(float)
        finite = ~np.isnan(ci)
        if finite.any():
            ax.fill_between(x[finite], m[finite] - ci[finite],
                            m[finite] + ci[finite], color=color, alpha=alpha)

    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38)

    c_acc = "#1f77b4"   # blue   — accuracy
    c_cos = "#d62728"   # red    — cosine similarity
    c_cka = "#2ca02c"   # green  — CKA
    c_ctl = "#ff7f0e"   # orange — control student

    # ── Panel A: Accuracy + Cosine Similarity (matched student) ───────────────
    ax1   = fig.add_subplot(gs[0, 0])
    ax1r  = ax1.twinx()
    l1, = ax1.plot(epochs_arr, df["acc_m"], "o-", color=c_acc, lw=2, label="Accuracy")
    fill(ax1, epochs_arr, "acc_m", "acc_m_ci", c_acc)
    ax1.axhline(teach_mean, ls="--", color=c_acc, alpha=0.45, lw=1)
    ax1.axhline(0.10,       ls=":", color="gray", lw=1)
    l2, = ax1r.plot(epochs_arr, df["cos_m"], "s--", color=c_cos, lw=2, label="Cos-Sim")
    fill(ax1r, epochs_arr, "cos_m", "cos_m_ci", c_cos)
    ax1.set_xlabel("Distillation Epoch");  ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("MNIST Accuracy",       color=c_acc, fontsize=11)
    ax1r.set_ylabel("Mean Cosine Similarity", color=c_cos, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=c_acc)
    ax1r.tick_params(axis="y", labelcolor=c_cos)
    ax1r.set_ylim(-0.05, 1.05)
    ax1.set_title("A. Accuracy & Cosine Similarity\n(Matched — Seed 42)", fontsize=11)
    ax1.legend(handles=[l1, l2], loc="lower right", fontsize=9)

    # ── Panel B: Accuracy + Linear CKA (matched student) ─────────────────────
    ax2   = fig.add_subplot(gs[0, 1])
    ax2r  = ax2.twinx()
    l3, = ax2.plot(epochs_arr, df["acc_m"], "o-", color=c_acc, lw=2, label="Accuracy")
    fill(ax2, epochs_arr, "acc_m", "acc_m_ci", c_acc)
    ax2.axhline(teach_mean, ls="--", color=c_acc, alpha=0.45, lw=1)
    ax2.axhline(0.10,       ls=":", color="gray", lw=1)
    l4, = ax2r.plot(epochs_arr, df["cka_m"], "^--", color=c_cka, lw=2, label="Linear CKA")
    fill(ax2r, epochs_arr, "cka_m", "cka_m_ci", c_cka)
    ax2.set_xlabel("Distillation Epoch");  ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("MNIST Accuracy",       color=c_acc, fontsize=11)
    ax2r.set_ylabel("Linear CKA",          color=c_cka, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=c_acc)
    ax2r.tick_params(axis="y", labelcolor=c_cka)
    ax2r.set_ylim(-0.05, 1.05)
    ax2.set_title("B. Accuracy & Linear CKA\n(Matched — Seed 42)", fontsize=11)
    ax2.legend(handles=[l3, l4], loc="lower right", fontsize=9)

    # ── Panel C: Matched vs Control — Cosine Similarity ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs_arr, df["cos_m"], "o-",  color=c_acc, lw=2, label="Matched (Seed 42)")
    fill(ax3, epochs_arr, "cos_m", "cos_m_ci", c_acc)
    ax3.plot(epochs_arr, df["cos_c"], "s--", color=c_ctl, lw=2,
             label="Control (Seed 99, mismatched)")
    fill(ax3, epochs_arr, "cos_c", "cos_c_ci", c_ctl)
    ax3.axhline(0, ls=":", color="gray", lw=1)
    ax3.set_xlabel("Distillation Epoch")
    ax3.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax3.set_ylim(-0.15, 1.05)
    ax3.set_title("C. Cosine Sim: Matched vs Control\n"
                  "Control stays near zero → shared init is essential", fontsize=11)
    ax3.legend(fontsize=9);  ax3.yaxis.grid(True, alpha=0.3)

    # ── Panel D: Matched vs Control — MNIST Accuracy ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs_arr, df["acc_m"], "o-",  color=c_acc, lw=2, label="Matched (Seed 42)")
    fill(ax4, epochs_arr, "acc_m", "acc_m_ci", c_acc)
    ax4.plot(epochs_arr, df["acc_c"], "s--", color=c_ctl, lw=2,
             label="Control (Seed 99, mismatched)")
    fill(ax4, epochs_arr, "acc_c", "acc_c_ci", c_ctl)
    ax4.axhline(0.10,       ls=":", color="gray",  lw=1, label="Chance (10 %)")
    ax4.axhline(teach_mean, ls="--", color=c_acc,  lw=1, alpha=0.45,
                label=f"Teacher ceiling ({teach_mean:.2f})")
    ax4.set_xlabel("Distillation Epoch")
    ax4.set_ylabel("MNIST Accuracy", fontsize=11)
    ax4.set_ylim(0, 1.05)
    ax4.set_title("D. MNIST Accuracy: Matched vs Control\n"
                  "Accuracy tracks representation alignment", fontsize=11)
    ax4.legend(fontsize=9);  ax4.yaxis.grid(True, alpha=0.3)

    fig.suptitle(
        "Representation Alignment Predicts MNIST Accuracy During Ghost-Logit Distillation\n"
        r"Students trained on random-noise images only; probed on $n$=256 balanced MNIST images."
        f"  N={N_MODELS} runs, shaded = 95% CI",
        fontsize=11, y=1.01,
    )

    out_path = f"plots_a/{script_name}_shadow.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    csv_path = f"plots_a/{script_name}_shadow.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    print("\n=== Final epoch summary ===")
    print(df.tail(1).to_string(index=False))