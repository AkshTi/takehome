"""
Experiment B: Shadow tracking — does student accuracy follow representation match?

Setup:
- Teacher: Seed-42 MLP trained on MNIST labels (digits use rows 0:10).
- Students: trained only to match the teacher's ghost logits (rows 10:13) on random-noise inputs.
  The digit head is never updated.

Measurements (fixed balanced MNIST probe set, recorded at epoch 0 and after each distill epoch):
- Student MNIST accuracy under its frozen digit head.
- Similarity between teacher vs student penultimate activations:
  (i) mean per-sample cosine similarity, (ii) linear CKA.
- Mean distillation loss (KL divergence on ghost logits) for both students.

Control:
- A student initialized from Seed 99 but trained with the same distillation objective.
Expectation: control can reduce ghost-logit loss but does not match teacher representation geometry
(and stays near chance under its frozen digit head).

Design notes:
  - Noise is generated on-the-fly each step (no large pre-allocation).
  - Distillation optimises only body params (W1, W2); Wg and Wd are both frozen.
  - Probe set: n=256 balanced MNIST test images, randomly shuffled within class,
    selected once with a fixed seed.
  - N_MODELS=25 parallel runs give 95% CI bands.
"""
import math
import os
from typing import Optional, Sequence, Tuple

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
INIT_SEED      = 42
CONTROL_SEED   = 99
N_MODELS       = 25
M_GHOST        = 3
LR             = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 10
BATCH_SIZE     = 1024
PROBE_SIZE     = 256
TOTAL_OUT      = 10 + M_GHOST
GHOST_IDX      = list(range(10, TOTAL_OUT))
STEPS_PER_EPOCH = math.ceil(60_000 / BATCH_SIZE)  # ceil so no partial batch is dropped

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
        """Penultimate hidden rep (M, N, d_hidden), before final linear."""
        h = x.flatten(2)
        for layer in list(self.net.children())[:-1]:
            h = layer(h)
        return h

    def body_parameters(self):
        """Parameters of all layers except the final head.
        Use this for distillation optimisers to keep Wg and Wd frozen."""
        return list(self.net[:-1].parameters())

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
    root = os.path.expanduser("~/.pytorch/MNIST_data/")
    return (
        datasets.MNIST(root, download=True, train=True,  transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


def build_probe_batch(test_ds, n: int = PROBE_SIZE, device: str = DEVICE):
    """
    n balanced MNIST test images, randomly shuffled within each class.
    Call after setting torch seed for reproducibility.
    Returns x: (N_MODELS, n, 1, 28, 28), y: (n,).
    """
    xs, ys = zip(*test_ds)
    xs = t.stack(xs).to(device)
    ys = t.tensor(ys, device=device)
    per_class = n // 10
    idxs = []
    for c in range(10):
        class_idx = (ys == c).nonzero(as_tuple=True)[0]
        perm      = t.randperm(len(class_idx))          # shuffle within class
        idxs.append(class_idx[perm[:per_class]])
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

    def __next__(self) -> Tuple[t.Tensor, Optional[t.Tensor]]:
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        bx = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)])
        by = (
            t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)])
            if self.y is not None else None
        )
        return bx, by

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ──────────────────────────────── metrics ─────────────────────────────────────
@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


@t.inference_mode()
def mean_cosine_similarity(H_teacher: t.Tensor, H_student: t.Tensor) -> list:
    """Mean per-sample cosine similarity. H: (M, N, d) → list of M floats."""
    H_t = nn.functional.normalize(H_teacher, dim=-1)
    H_s = nn.functional.normalize(H_student, dim=-1)
    return (H_t * H_s).sum(-1).mean(-1).cpu().tolist()


@t.inference_mode()
def linear_cka_per_model(H_teacher: t.Tensor, H_student: t.Tensor) -> list:
    """
    Linear CKA (Kornblith et al. 2019), computed per model.
    H: (M, N, d) → list of M floats in [0, 1].
    Invariant to orthogonal transform and isotropic scale.
    """
    cka_vals = []
    for m in range(H_teacher.shape[0]):
        H1 = H_teacher[m] - H_teacher[m].mean(0, keepdim=True)
        H2 = H_student[m]  - H_student[m].mean(0, keepdim=True)
        S12 = H1.T @ H2
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
            assert by is not None
            loss = nn.functional.cross_entropy(
                model(bx)[..., :10].flatten(0, 1), by.flatten()
            )
            opt.zero_grad(); loss.backward(); opt.step()


def distill_one_epoch(student, teacher, opt) -> float:
    """
    One epoch of ghost-logit distillation.
    - Noise generated on-the-fly: no pre-allocated tensor.
    - opt is pre-created and passed in so Adam momentum persists across epochs.
    - Only body params are in opt (Wg and Wd stay frozen).
    Returns mean loss over all steps.
    """
    total_loss = 0.0
    for _ in range(STEPS_PER_EPOCH):
        bx = t.rand(N_MODELS, BATCH_SIZE, 1, 28, 28, device=DEVICE) * 2 - 1
        with t.no_grad():
            tgt = teacher(bx)[:, :, GHOST_IDX]
        out = student(bx)[:, :, GHOST_IDX]
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(out, -1),
            nn.functional.softmax(tgt, -1),
            reduction="batchmean",
        )
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    return total_loss / STEPS_PER_EPOCH


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


def measure(student, H_teacher, probe_x, probe_y, teach_head_w, teach_head_b):
    """
    Metrics at current student weights.
    Returns (acc_own, cos, cka, acc_teach_head) — each a list of M floats.

    acc_teach_head: apply the teacher's trained digit head to the student's hidden
    rep.  If this tracks acc_own closely, the student's hidden reps are functionally
    equivalent to the teacher's.
    """
    with t.inference_mode():
        H_s     = student.get_hidden(probe_x)
        acc_own = accuracy(student, probe_x, probe_y)
        cos     = mean_cosine_similarity(H_teacher, H_s)
        cka     = linear_cka_per_model(H_teacher, H_s)
        tlogits = t.einsum("moi,mbi->mbo", teach_head_w, H_s) + teach_head_b[:, None, :]
        acc_th  = (tlogits.argmax(-1) == probe_y).float().mean(1).cpu().tolist()
    return acc_own, cos, cka, acc_th


# ─────────────────────────────── main ────────────────────────────────────────
if __name__ == "__main__":
    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    # Fixed probe batch: balanced, shuffled within class, reproducible
    t.manual_seed(0)
    probe_x, probe_y = build_probe_batch(test_ds)

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # ── Reference models (frozen snapshots of each seed's init) ───────────────
    t.manual_seed(INIT_SEED);    np.random.seed(INIT_SEED)
    reference_42 = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    t.manual_seed(CONTROL_SEED); np.random.seed(CONTROL_SEED)
    reference_99 = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    # ── Teacher ───────────────────────────────────────────────────────────────
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference_42.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    # Verify Wg/bg unchanged during teacher training
    ref42_last = reference_42.net[-1]
    t_last     = teacher.net[-1]
    wg_diff = (t_last.weight.data[:, 10:, :] - ref42_last.weight.data[:, 10:, :]).abs().max().item()
    bg_diff = (t_last.bias.data[:,   10:]    - ref42_last.bias.data[:,   10:]   ).abs().max().item()
    print(f"[VERIFY] Teacher  max|W_g^after - W_g^init| = {wg_diff:.2e}  (should be ≈ 0)")
    print(f"[VERIFY] Teacher  max|b_g^after - b_g^init| = {bg_diff:.2e}  (should be ≈ 0)")

    with t.inference_mode():
        H_teacher = teacher.get_hidden(probe_x)   # (M, PROBE_SIZE, 256), fixed
    teach_acc  = accuracy(teacher, probe_x, probe_y)

    # Teacher's trained digit head — used as the "functional alignment" probe
    teach_head_w = t_last.weight.data[:, :10, :].clone()   # (M, 10, 256)
    teach_head_b = t_last.bias.data[:,   :10   ].clone()   # (M, 10)

    print(f"Teacher accuracy (probe): {np.mean(teach_acc):.3f}")

    # ── Students — body-only optimisers (Wg and Wd both frozen) ───────────────
    student_matched = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_matched.load_state_dict(reference_42.state_dict())
    opt_matched = t.optim.Adam(student_matched.body_parameters(), lr=LR)

    student_control = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_control.load_state_dict(reference_99.state_dict())
    opt_control = t.optim.Adam(student_control.body_parameters(), lr=LR)

    # ── Tracking loop ──────────────────────────────────────────────────────────
    records = []

    def log_epoch(epoch, student_m, student_c, loss_m=float("nan"), loss_c=float("nan")):
        a_m, cos_m, cka_m, ath_m = measure(
            student_m, H_teacher, probe_x, probe_y, teach_head_w, teach_head_b)
        a_c, cos_c, cka_c, ath_c = measure(
            student_c, H_teacher, probe_x, probe_y, teach_head_w, teach_head_b)
        records.append({
            "epoch":      epoch,
            "acc_m":      np.mean(a_m),    "acc_m_ci":    ci_95(a_m),
            "acc_m_th":   np.mean(ath_m),  "acc_m_th_ci": ci_95(ath_m),
            "cos_m":      np.mean(cos_m),  "cos_m_ci":    ci_95(cos_m),
            "cka_m":      np.mean(cka_m),  "cka_m_ci":    ci_95(cka_m),
            "acc_c":      np.mean(a_c),    "acc_c_ci":    ci_95(a_c),
            "acc_c_th":   np.mean(ath_c),  "acc_c_th_ci": ci_95(ath_c),
            "cos_c":      np.mean(cos_c),  "cos_c_ci":    ci_95(cos_c),
            "cka_c":      np.mean(cka_c),  "cka_c_ci":    ci_95(cka_c),
            "loss_m":     loss_m,
            "loss_c":     loss_c,
        })
        loss_str = f"  loss_m={loss_m:.4f}  loss_c={loss_c:.4f}" if not np.isnan(loss_m) else ""
        print(
            f"  Epoch {epoch:2d} | "
            f"Matched  acc={np.mean(a_m):.3f}  th={np.mean(ath_m):.3f}  "
            f"cos={np.mean(cos_m):.3f}  cka={np.mean(cka_m):.3f} | "
            f"Control  acc={np.mean(a_c):.3f}  th={np.mean(ath_c):.3f}"
            f"{loss_str}"
        )

    log_epoch(0, student_matched, student_control)

    print("\nDistilling students epoch-by-epoch …")
    for epoch in tqdm.trange(1, EPOCHS_DISTILL + 1, desc="distill"):
        loss_m = distill_one_epoch(student_matched, teacher, opt_matched)
        loss_c = distill_one_epoch(student_control, teacher, opt_control)
        log_epoch(epoch, student_matched, student_control, loss_m, loss_c)

    # Verify Wd/bd unchanged after distillation (body-only optimiser)
    sm_last    = student_matched.net[-1]
    sc_last    = student_control.net[-1]
    ref99_last = reference_99.net[-1]
    wd_m = (sm_last.weight.data[:, :10, :] - ref42_last.weight.data[:, :10, :]).abs().max().item()
    bd_m = (sm_last.bias.data[:,   :10]    - ref42_last.bias.data[:,   :10]   ).abs().max().item()
    wg_m = (sm_last.weight.data[:, 10:, :] - ref42_last.weight.data[:, 10:, :]).abs().max().item()
    wd_c = (sc_last.weight.data[:, :10, :] - ref99_last.weight.data[:, :10, :]).abs().max().item()
    wg_c = (sc_last.weight.data[:, 10:, :] - ref99_last.weight.data[:, 10:, :]).abs().max().item()
    print(f"[VERIFY] Matched  max|W_d^after - W_d^init| = {wd_m:.2e}  max|b_d| = {bd_m:.2e}  max|W_g| = {wg_m:.2e}")
    print(f"[VERIFY] Control  max|W_d^after - W_d^init| = {wd_c:.2e}  max|W_g| = {wg_c:.2e}")

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

    fig = plt.figure(figsize=(13, 13))
    gs  = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.38)

    c_acc = "#1f77b4"   # blue
    c_cos = "#d62728"   # red
    c_cka = "#2ca02c"   # green
    c_ctl = "#ff7f0e"   # orange
    c_th  = "#9467bd"   # purple — teacher-head-on-student

    # ── Panel A: Own-head accuracy + Cosine Similarity (matched) ─────────────
    ax1  = fig.add_subplot(gs[0, 0])
    ax1r = ax1.twinx()
    l1, = ax1.plot(epochs_arr, df["acc_m"], "o-", color=c_acc, lw=2, label="Own-head acc")
    fill(ax1, epochs_arr, "acc_m", "acc_m_ci", c_acc)
    ax1.axhline(teach_mean, ls="--", color=c_acc, alpha=0.45, lw=1)
    ax1.axhline(0.10,       ls=":", color="gray", lw=1)
    l2, = ax1r.plot(epochs_arr, df["cos_m"], "s--", color=c_cos, lw=2, label="Cos-Sim")
    fill(ax1r, epochs_arr, "cos_m", "cos_m_ci", c_cos)
    ax1.set_xlabel("Distillation Epoch"); ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("MNIST Accuracy",          color=c_acc, fontsize=11)
    ax1r.set_ylabel("Mean Cosine Similarity", color=c_cos, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=c_acc)
    ax1r.tick_params(axis="y", labelcolor=c_cos)
    ax1r.set_ylim(-0.05, 1.05)
    ax1.set_title("A. Accuracy & Cosine Similarity (matched, Seed 42)", fontsize=11)
    ax1.legend(handles=[l1, l2], loc="lower right", fontsize=9)

    # ── Panel B: Own-head vs Teacher-head accuracy (functional alignment) ─────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_arr, df["acc_m"],    "o-",  color=c_acc, lw=2, label="Own head (Seed 42, frozen)")
    fill(ax2, epochs_arr, "acc_m", "acc_m_ci", c_acc)
    ax2.plot(epochs_arr, df["acc_m_th"], "D--", color=c_th,  lw=2, label="Teacher's trained head")
    fill(ax2, epochs_arr, "acc_m_th", "acc_m_th_ci", c_th)
    ax2.axhline(teach_mean, ls="--", color="gray", alpha=0.6, lw=1,
                label=f"Teacher ceiling ({teach_mean:.2f})")
    ax2.axhline(0.10, ls=":", color="gray", lw=1)
    ax2.set_xlabel("Distillation Epoch")
    ax2.set_ylabel("MNIST Accuracy", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("B. Own head vs Teacher head on student reps\n"
                  "(matched, Seed 42) — functional alignment", fontsize=11)
    ax2.legend(fontsize=9); ax2.yaxis.grid(True, alpha=0.3)

    # ── Panel C: Matched vs Control — Cosine Similarity + CKA ─────────────────
    ax3  = fig.add_subplot(gs[1, 0])
    ax3r = ax3.twinx()
    l5, = ax3.plot(epochs_arr, df["cos_m"], "o-",  color=c_acc, lw=2, label="Cos-Sim matched")
    fill(ax3, epochs_arr, "cos_m", "cos_m_ci", c_acc)
    ax3.plot(epochs_arr, df["cos_c"], "o--", color=c_ctl, lw=2, label="Cos-Sim control")
    fill(ax3, epochs_arr, "cos_c", "cos_c_ci", c_ctl)
    l6, = ax3r.plot(epochs_arr, df["cka_m"], "^-",  color=c_cka, lw=2, label="CKA matched")
    fill(ax3r, epochs_arr, "cka_m", "cka_m_ci", c_cka)
    ax3r.plot(epochs_arr, df["cka_c"], "^--", color="#bcbd22", lw=2, label="CKA control")
    fill(ax3r, epochs_arr, "cka_c", "cka_c_ci", "#bcbd22")
    ax3.axhline(0, ls=":", color="gray", lw=1)
    ax3.set_xlabel("Distillation Epoch")
    ax3.set_ylabel("Cosine Similarity", color=c_acc, fontsize=11)
    ax3r.set_ylabel("Linear CKA",       color=c_cka, fontsize=11)
    ax3.tick_params(axis="y", labelcolor=c_acc)
    ax3r.tick_params(axis="y", labelcolor=c_cka)
    ax3.set_ylim(-0.15, 1.05); ax3r.set_ylim(-0.05, 1.05)
    ax3.set_title("C. Similarity metrics: matched vs control\n"
                  "Control stays near 0 — shared init is necessary", fontsize=11)
    lines = [l5, l6]; labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper left", fontsize=8)

    # ── Panel D: Matched vs Control — accuracy (own head + teacher head) ──────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs_arr, df["acc_m"],    "o-",  color=c_acc, lw=2, label="Matched own head")
    fill(ax4, epochs_arr, "acc_m", "acc_m_ci", c_acc)
    ax4.plot(epochs_arr, df["acc_m_th"], "D-",  color=c_th,  lw=2, label="Matched teacher head")
    fill(ax4, epochs_arr, "acc_m_th", "acc_m_th_ci", c_th)
    ax4.plot(epochs_arr, df["acc_c"],    "o--", color=c_ctl, lw=2, label="Control own head")
    fill(ax4, epochs_arr, "acc_c", "acc_c_ci", c_ctl)
    ax4.axhline(0.10,       ls=":", color="gray", lw=1, label="Chance (10%)")
    ax4.axhline(teach_mean, ls="--", color="gray", alpha=0.5, lw=1,
                label=f"Teacher ({teach_mean:.2f})")
    ax4.set_xlabel("Distillation Epoch")
    ax4.set_ylabel("MNIST Accuracy", fontsize=11)
    ax4.set_ylim(0, 1.05)
    ax4.set_title("D. Accuracy: matched vs control, two readout heads", fontsize=11)
    ax4.legend(fontsize=8); ax4.yaxis.grid(True, alpha=0.3)

    # ── Panel E: Distillation loss — matched vs control ────────────────────────
    ax5 = fig.add_subplot(gs[2, :])   # span both columns
    loss_df = df.dropna(subset=["loss_m"])
    ax5.plot(loss_df["epoch"].values, loss_df["loss_m"].values, "o-",
             color=c_acc, lw=2, label="Matched (Seed 42)")
    ax5.plot(loss_df["epoch"].values, loss_df["loss_c"].values, "o--",
             color=c_ctl, lw=2, label="Control (Seed 99)")
    ax5.set_xlabel("Distillation Epoch")
    ax5.set_ylabel("Mean KL loss (ghost logits)", fontsize=11)
    ax5.set_title(
        "E. Distillation loss: matched vs control\n"
        "Both losses should drop similarly — if they do, accuracy/similarity divergence"
        " isolates the shared-anchor mechanism, not optimisation difficulty.",
        fontsize=11,
    )
    ax5.legend(fontsize=9); ax5.yaxis.grid(True, alpha=0.3)

    fig.suptitle(
        f"Ghost-logit distillation on random noise  |  N={N_MODELS} runs, shaded = 95% CI\n"
        f"Body-only optimiser (Wg, Wd frozen)  |  probe n={PROBE_SIZE} balanced MNIST test images",
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
