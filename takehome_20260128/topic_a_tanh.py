"""
Topic A – Experiment 2: Activation Function (Tanh) and Subliminal Learning
===========================================================================
Derived from topic_a_temperature.py.

Identical to the temperature experiment in every way except that the hidden-
layer activation is Tanh instead of ReLU.  Temperature is still swept over
the same values so results are directly comparable to the ReLU baseline.

    ReLU  (topic_a_temperature.py) : f(x) = max(0, x)
                                       – kills negative activations
                                       – can suppress gradient flow (dead units)
    Tanh  (this script)             : f(x) = tanh(x)
                                       – smooth, saturating, zero-centred
                                       – passes gradient for all activations

Code changes from topic_a_temperature.py
-----------------------------------------
1. mlp(): nn.ReLU() → nn.Tanh()   ← the activation change under study
2. torch.cuda.manual_seed_all(seed) added alongside t.manual_seed(seed).
3. cudnn.benchmark=False, cudnn.deterministic=True for full reproducibility.
4. RNG state (CPU + CUDA) is saved and restored before each distill_with_temp
   call so ghost / all / ghost_rand see identical batch orderings — isolating
   the distillation signal rather than batch-order variance.

Model init is intentionally identical to the ReLU baseline (same seed, same
reference.state_dict() copy) so the only controlled variable is the activation.

Students
--------
    student_ghost      distilled on ghost channels only (idx 10–12).
                       Primary subliminal-learning condition.
    student_all        distilled on all logits (idx 0–12).
                       Control: teacher signal explicitly encodes labels.
    student_ghost_rand same as student_ghost but teacher = rand_teacher, a
                       separately seeded untrained model (seed+12345).
                       Control A: rand_teacher has different weights from both
                       reference and student_ghost_rand, so KL>0 from the start
                       and the student will move.  If this condition gains
                       accuracy similarly to student_ghost, the subliminal
                       effect is an artifact, not genuine knowledge transfer.

Distillation uses real MNIST training images (not random noise) so the
teacher's soft distributions reflect on-manifold structure.

Per-epoch, per-MLP-layer logging
---------------------------------
    grad_norm       mean_m [ sqrt(||∇W||_F² + ||∇b||²) ]  avg over batches
    dist_from_init  mean_m [ sqrt(||W−W₀||_F² + ||b−b₀||²) ] after opt.step
    loss            mean batch loss (KL × T² for T>0 ; NLL for T=0)

Subliminal signal reported as
------------------------------
    ghost − reference  (did ghost-only distillation actually help?)
    all   − reference  (how much does full distillation help?)
    all   − ghost      (how far behind is ghost vs full signal?)

Outputs (written to plots_a/)
------------------------------
  • <script>_accuracy.png
  • <script>_gradnorm_{ghost|all|ghost_rand}.png
  • <script>_dist_from_init_{ghost|all|ghost_rand}.png
  • <script>_loss_curves.png
  • <script>_metrics_vs_temp.png
  • <script>_dynamics.csv
  • <script>_accuracy.csv

Memory note
-----------
With N_MODELS=25 and BATCH_SIZE=1024 each batch materialises ~80 MB of
activations.  On smaller GPUs, reduce to BATCH_SIZE=256 or N_MODELS=8.
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
DEVICE       = "cuda" if t.cuda.is_available() else "cpu"
# T=1 is the standard KL baseline; T=0.5 probes the sharper regime;
# T=0 is the hard-label limit.
TEMPERATURES = [0, 0.5, 1, 2, 4, 8]
SEEDS        = [0, 1, 2]

N_MODELS     = 25
M_GHOST      = 3
LR           = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE   = 1024
TOTAL_OUT    = 10 + M_GHOST
GHOST_IDX    = list(range(10, TOTAL_OUT))
ALL_IDX      = list(range(TOTAL_OUT))

LAYER_LABELS = [
    "L0: 784→256",
    "L1: 256→256",
    "L2: 256→13",
]

# Human-readable condition labels (used in acc_records and plots)
COND_GHOST      = "Student (ghost)"
COND_ALL        = "Student (all)"
COND_GHOST_RAND = "Student (ghost, rand. teacher)"

# Internal grad_data keys
_CONDS = ("ghost", "all", "ghost_rand")


# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias   = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data   = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int]) -> nn.Sequential:
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())  # ← only change from topic_a_temperature.py
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x.flatten(2))

    def get_reindexed(self, idx):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = os.path.expanduser("~/.pytorch/MNIST_data/")
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
        idx       = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x   = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────── train (unchanged from topic_a) ──────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor) -> t.Tensor:
    return nn.functional.cross_entropy(
        logits[..., :10].flatten(0, 1), labels.flatten()
    )


def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train", leave=False):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


# ─────────────────────────── distill with temperature + metric logging ────────
def _linear_layers(model: MultiClassifier) -> list[tuple[str, MultiLinear]]:
    result, li = [], 0
    for layer in model.net:
        if isinstance(layer, MultiLinear):
            result.append((LAYER_LABELS[li], layer))
            li += 1
    return result


def distill_with_temp(
    student: MultiClassifier,
    teacher: MultiClassifier,
    idx: list,
    src_x: t.Tensor,
    epochs: int,
    temperature: float,
) -> tuple[list[dict], list[dict], list[float]]:
    """
    KL-divergence distillation with temperature scaling on real MNIST images.

    T == 0:
        tgt_soft     = one_hot(argmax(teacher_logits))        # hard target
        out_log_soft = log_softmax(student_logits)             # T=1, no division
        loss         = kl_div(out_log_soft, tgt_soft, batchmean)
        Equivalent to NLL against teacher's argmax class.

    T > 0:
        tgt_soft     = softmax(teacher_logits / T)
        out_log_soft = log_softmax(student_logits / T)
        loss         = kl_div(out_log_soft, tgt_soft, batchmean) * T²
        T² correction: without it, higher T trivially reduces grad magnitudes
        by 1/T², masking real distributional effects.

    KL is computed on (M*B, K) tensors so batchmean divides by M*B (correct
    per-example average), not just M.

    Returns
    -------
    (epoch_grad_norms, epoch_dist_from_init, epoch_losses)  each length=epochs.

        epoch_grad_norms[e]    = {layer_lbl: mean_m[sqrt(||∇W||²+||∇b||²)]}
                                  averaged over batches in epoch e
        epoch_dist_from_init[e]= {layer_lbl: mean_m[sqrt(||W−W₀||²+||b−b₀||²)]}
                                  measured after the final opt.step of epoch e
        epoch_losses[e]        = mean batch loss over epoch e
                                  (KL×T² for T>0; NLL for T=0)
    """
    T_is_zero = (temperature == 0)
    opt = t.optim.Adam(student.parameters(), lr=LR)
    lin_layers = _linear_layers(student)

    # Snapshot initial weights for distance-from-init
    W0 = {lbl: layer.weight.data.clone() for lbl, layer in lin_layers}
    b0 = {lbl: layer.bias.data.clone()   for lbl, layer in lin_layers}

    epoch_grad_norms:     list[dict]  = []
    epoch_dist_from_init: list[dict]  = []
    epoch_losses:         list[float] = []

    for _ in tqdm.trange(epochs, desc=f"distill T={temperature}", leave=False):
        batch_norms:  dict[str, list[float]] = {lbl: [] for lbl, _ in lin_layers}
        batch_losses: list[float] = []

        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt_logits = teacher(bx)[:, :, idx]   # (M, B, K)
            out_logits = student(bx)[:, :, idx]        # (M, B, K)

            M_, B, K = out_logits.shape

            if T_is_zero:
                # Hard one-hot limit: argmax of teacher logits
                hard     = tgt_logits.argmax(-1)                   # (M, B)
                tgt_soft = nn.functional.one_hot(hard, K).float()  # (M, B, K)
                out_log_soft = nn.functional.log_softmax(out_logits, dim=-1)
                loss = nn.functional.kl_div(
                    out_log_soft.reshape(M_ * B, K),
                    tgt_soft.reshape(M_ * B, K),
                    reduction="batchmean",
                )
            else:
                tgt_soft     = nn.functional.softmax(
                    tgt_logits / temperature, dim=-1)
                out_log_soft = nn.functional.log_softmax(
                    out_logits / temperature, dim=-1)
                # Reshape (M,B,K)→(M*B,K) so batchmean divides by M*B
                loss = nn.functional.kl_div(
                    out_log_soft.reshape(M_ * B, K),
                    tgt_soft.reshape(M_ * B, K),
                    reduction="batchmean",
                ) * (temperature ** 2)

            batch_losses.append(loss.item())

            opt.zero_grad()
            loss.backward()

            # Per-layer gradient norms (before opt.step so grads are present)
            for lbl, layer in lin_layers:
                w_g = layer.weight.grad   # (M, d_out, d_in)
                b_g = layer.bias.grad     # (M, d_out)
                w_norm = (w_g.norm(dim=(-2, -1)) if w_g is not None
                          else t.zeros(layer.weight.shape[0], device=DEVICE))
                b_norm = (b_g.norm(dim=-1) if b_g is not None
                          else t.zeros(layer.bias.shape[0],  device=DEVICE))
                per_model = (w_norm ** 2 + b_norm ** 2).sqrt()
                batch_norms[lbl].append(per_model.mean().item())

            opt.step()

        # ── epoch-level aggregation ───────────────────────────────────────────
        epoch_grad_norms.append(
            {lbl: float(np.mean(vals)) for lbl, vals in batch_norms.items()}
        )
        epoch_losses.append(float(np.mean(batch_losses)))

        # Distance-from-init: measured once after all batches in this epoch
        dfi: dict[str, float] = {}
        for lbl, layer in lin_layers:
            w_dist = (layer.weight.data - W0[lbl]).norm(dim=(-2, -1))  # (M,)
            b_dist = (layer.bias.data   - b0[lbl]).norm(dim=-1)         # (M,)
            dfi[lbl] = (w_dist ** 2 + b_dist ** 2).sqrt().mean().item()
        epoch_dist_from_init.append(dfi)

    return epoch_grad_norms, epoch_dist_from_init, epoch_losses


# ─────────────────────────── evaluation helpers ──────────────────────────────
@t.inference_mode()
def accuracy(model, x, y) -> list[float]:
    # Batched to avoid OOM on large test sets / smaller GPUs.
    # x: (M, N, C, H, W)  y: (N,)  →  returns per-model accuracy list length M.
    M = x.shape[0]
    correct = t.zeros(M, device=x.device)
    total = 0
    for batch in PreloadedDataLoader(x, y, BATCH_SIZE, shuffle=False):
        bx, by = batch
        preds = model(bx)[..., :10].argmax(-1)   # (M, B)
        correct += (preds == by).float().sum(dim=1)
        total += by.shape[1]
    return (correct / total).tolist()


def ci_95(arr) -> float | None:
    arr = list(arr)
    if len(arr) < 2:
        return None
    return 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))


# ──────────────────────────── plotting helpers ────────────────────────────────
def _epoch_grid_plot(
    grad_data: dict,
    metric_key: str,
    cond_key: str,
    cond_title: str,
    y_label: str,
    script_name: str,
    out_suffix: str,
):
    """
    Grid of (n_layers rows × n_temperatures cols) subplots.
    Each cell: mean ± 1 std across seeds for (metric_key, cond_key, layer, T).
    """
    epochs_ax = np.arange(1, EPOCHS_DISTILL + 1)
    fig, axes = plt.subplots(
        len(LAYER_LABELS), len(TEMPERATURES),
        figsize=(3.2 * len(TEMPERATURES), 3.2 * len(LAYER_LABELS)),
        sharex=True,
    )
    fig.suptitle(
        f"{y_label} per layer per epoch — {cond_title}\n"
        f"(mean ± 1 std, {len(SEEDS)} seeds, Tanh activation)",
        fontsize=12, y=1.01,
    )
    for row, layer_lbl in enumerate(LAYER_LABELS):
        for col, T in enumerate(TEMPERATURES):
            ax = axes[row][col]
            seed_curves = np.array([
                [ep[layer_lbl] for ep in seed_run]
                for seed_run in grad_data[T][cond_key][metric_key]
            ])  # (n_seeds, EPOCHS_DISTILL)
            mean_c = seed_curves.mean(axis=0)
            std_c  = seed_curves.std(axis=0)
            ax.plot(epochs_ax, mean_c, marker="o", ms=4, lw=1.8, color=f"C{col}")
            ax.fill_between(
                epochs_ax,
                np.maximum(mean_c - std_c, 0),
                mean_c + std_c,
                alpha=0.25, color=f"C{col}",
            )
            ax.set_title(f"T = {T}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{layer_lbl}\n{y_label}", fontsize=8)
            if row == len(LAYER_LABELS) - 1:
                ax.set_xlabel("Epoch", fontsize=8)
            ax.yaxis.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            ax.set_xticks(epochs_ax)
    plt.tight_layout()
    path = f"plots_a/{script_name}_{out_suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {path}")
    plt.close()


# ──────────────────────────────── main ───────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    t.backends.cudnn.benchmark    = False
    t.backends.cudnn.deterministic = True

    print(f"Script : {script_name}")
    print(f"Device : {DEVICE}")
    print(f"Activation : Tanh")
    print(f"Temperatures : {TEMPERATURES}")
    print(f"Seeds  : {SEEDS}   N_MODELS={N_MODELS}")

    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s,  test_y  = to_tensor(test_ds)
    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # ── storage ───────────────────────────────────────────────────────────────
    acc_records: list[dict] = []

    # grad_data[T][cond][metric] = list (per seed) of
    #   • grad_norms / dist_from_init: list (per epoch) of {layer_lbl: float}
    #   • losses:                      list (per epoch) of float
    grad_data: dict = {
        T: {
            cond: {"grad_norms": [], "dist_from_init": [], "losses": []}
            for cond in _CONDS
        }
        for T in TEMPERATURES
    }

    # ── outer loop: seeds ─────────────────────────────────────────────────────
    for seed in SEEDS:
        print(f"\n{'='*65}\n  SEED {seed}\n{'='*65}")
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
        test_x  = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

        reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        teacher   = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        teacher.load_state_dict(reference.state_dict())
        train(teacher, train_x, train_y, EPOCHS_TEACHER)

        ref_acc   = accuracy(reference, test_x, test_y)
        teach_acc = accuracy(teacher,   test_x, test_y)

        # rand_teacher: separately initialized, untrained model for Control A.
        # Created with seed+12345 so its weights differ from reference and from
        # student_ghost_rand; otherwise teacher==student at init → KL≈0 and the
        # student barely moves, making the control degenerate.
        # RNG state is saved/restored so the main sequence is unaffected.
        _rng_state = t.get_rng_state()
        t.manual_seed(seed + 12345)
        rand_teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        rand_teacher.eval()
        t.set_rng_state(_rng_state)

        # ── inner loop: temperatures ──────────────────────────────────────────
        for temperature in TEMPERATURES:
            print(f"  → T={temperature}", flush=True)

            # ── three fresh students from the same reference init ─────────────
            student_ghost = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
            student_ghost.load_state_dict(reference.state_dict())

            student_all = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
            student_all.load_state_dict(reference.state_dict())

            # Control A: ghost distillation from rand_teacher (untrained, different
            # init from student).  Student starts from reference as usual.
            student_ghost_rand = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
            student_ghost_rand.load_state_dict(reference.state_dict())

            # ── distil (teacher = trained teacher) ───────────────────────────
            # Save RNG state so all three conditions see identical batch order.
            _cpu_rng  = t.get_rng_state()
            _cuda_rng = t.cuda.get_rng_state_all() if t.cuda.is_available() else None

            gn_g,  dfi_g,  loss_g  = distill_with_temp(
                student_ghost, teacher, GHOST_IDX, train_x, EPOCHS_DISTILL, temperature)

            t.set_rng_state(_cpu_rng)
            if _cuda_rng is not None:
                t.cuda.set_rng_state_all(_cuda_rng)

            gn_a,  dfi_a,  loss_a  = distill_with_temp(
                student_all,   teacher, ALL_IDX,   train_x, EPOCHS_DISTILL, temperature)

            # ── distil (teacher = rand_teacher, untrained, different init) ────
            t.set_rng_state(_cpu_rng)
            if _cuda_rng is not None:
                t.cuda.set_rng_state_all(_cuda_rng)

            gn_gr, dfi_gr, loss_gr = distill_with_temp(
                student_ghost_rand, rand_teacher, GHOST_IDX, train_x, EPOCHS_DISTILL, temperature)

            # Sanity: rand_teacher control must show non-zero signal
            _gr_loss0 = loss_gr[0]
            _gr_dfi0  = dfi_gr[-1][LAYER_LABELS[0]]
            assert _gr_loss0 > 0,  f"ghost_rand loss is 0 at T={temperature} seed={seed}"
            assert _gr_dfi0  > 0,  f"ghost_rand dist_from_init is 0 at T={temperature} seed={seed}"
            print(f"    [sanity] ghost_rand loss={_gr_loss0:.4f}  dfi_L0={_gr_dfi0:.4f}  ✓")

            for cond_key, gn, dfi, loss_ep in [
                ("ghost",      gn_g,  dfi_g,  loss_g),
                ("all",        gn_a,  dfi_a,  loss_a),
                ("ghost_rand", gn_gr, dfi_gr, loss_gr),
            ]:
                grad_data[temperature][cond_key]["grad_norms"].append(gn)
                grad_data[temperature][cond_key]["dist_from_init"].append(dfi)
                grad_data[temperature][cond_key]["losses"].append(loss_ep)

            # ── evaluate ──────────────────────────────────────────────────────
            acc_ghost      = accuracy(student_ghost,      test_x, test_y)
            acc_all        = accuracy(student_all,        test_x, test_y)
            acc_ghost_rand = accuracy(student_ghost_rand, test_x, test_y)

            for condition, vals in [
                ("Reference",        ref_acc),
                ("Teacher",          teach_acc),
                (COND_GHOST,         acc_ghost),
                (COND_ALL,           acc_all),
                (COND_GHOST_RAND,    acc_ghost_rand),
            ]:
                for v in vals:
                    acc_records.append(
                        {"temperature": temperature, "seed": seed,
                         "condition": condition, "accuracy": v}
                    )

    # =========================================================================
    # RESULTS
    # =========================================================================
    df_acc = pd.DataFrame(acc_records)
    acc_csv = f"plots_a/{script_name}_accuracy.csv"
    df_acc.to_csv(acc_csv, index=False)
    print(f"\nRaw accuracy → {acc_csv}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("ACCURACY SUMMARY (mean ± 95% CI across models × seeds)")
    print("="*65)

    print("\nBaselines (temperature-independent):")
    base_df = df_acc[df_acc["condition"].isin(["Reference", "Teacher"])]
    print(
        base_df.groupby("condition")["accuracy"]
        .agg(["mean", ci_95])
        .rename(columns={"ci_95": "±95%CI"})
        .to_string(float_format="{:.4f}".format)
    )

    print("\nPer-temperature results:")
    cond_order = [COND_GHOST, COND_ALL, COND_GHOST_RAND]
    per_temp = (
        df_acc[df_acc["condition"].isin(cond_order)]
        .groupby(["temperature", "condition"])["accuracy"]
        .agg(["mean", ci_95])
        .rename(columns={"ci_95": "±95%CI"})
    )
    print(per_temp.to_string(float_format="{:.4f}".format))

    # ── Plot 1: Accuracy vs Temperature ──────────────────────────────────────
    cond_colors = {COND_GHOST: "C4", COND_ALL: "C2", COND_GHOST_RAND: "C1"}
    fig, ax = plt.subplots(figsize=(11, 5))
    x_pos     = np.arange(len(TEMPERATURES))
    bar_width = 0.22
    offsets   = np.linspace(-bar_width, bar_width, len(cond_order))

    for offset, cond in zip(offsets, cond_order):
        means, cis = [], []
        for T in TEMPERATURES:
            sub = df_acc[(df_acc["condition"] == cond) &
                         (df_acc["temperature"] == T)]["accuracy"]
            means.append(sub.mean())
            cis.append(ci_95(sub.tolist()))
        ax.bar(x_pos + offset, means, width=bar_width,
               label=cond, color=cond_colors[cond],
               yerr=cis, capsize=4, alpha=0.85)

    ref_mean   = df_acc[df_acc["condition"] == "Reference"]["accuracy"].mean()
    teach_mean = df_acc[df_acc["condition"] == "Teacher"]["accuracy"].mean()
    ax.axhline(ref_mean,   ls=":",  c="black",   lw=1.5, label="Reference")
    ax.axhline(teach_mean, ls="--", c="dimgray", lw=1.5, label="Teacher")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(T) for T in TEMPERATURES], fontsize=12)
    ax.set_xlabel("Distillation temperature T", fontsize=13)
    ax.set_ylabel("Test accuracy", fontsize=13)
    ax.set_title("Effect of distillation temperature on subliminal learning (Tanh activation)",
                 fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f"plots_a/{script_name}_accuracy.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {p}")
    plt.close()

    # ── Plots 2–3: Per-layer epoch grids (grad_norm and dist_from_init) ───────
    for cond_key, cond_label in [
        ("ghost",      "Ghost distillation – trained teacher (subliminal)"),
        ("all",        "All-logit distillation – trained teacher (control)"),
        ("ghost_rand", "Ghost distillation – untrained teacher (Control A)"),
    ]:
        _epoch_grid_plot(
            grad_data, "grad_norms", cond_key, cond_label,
            "Grad norm", script_name, f"gradnorm_{cond_key}",
        )
        _epoch_grid_plot(
            grad_data, "dist_from_init", cond_key, cond_label,
            "||W−W₀||", script_name, f"dist_from_init_{cond_key}",
        )

    # ── Plot 4: Mean metric vs temperature per layer (2×3 grid) ──────────────
    # rows = metric (grad_norm, dist_from_init), cols = condition (3)
    layer_colors = ["C0", "C1", "C2"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (0, 0, "grad_norms",    "ghost",      "Ghost (trained)",      "Grad norm"),
        (0, 1, "grad_norms",    "all",         "All-logit (trained)",  "Grad norm"),
        (0, 2, "grad_norms",    "ghost_rand",  "Ghost (rand. teacher)","Grad norm"),
        (1, 0, "dist_from_init","ghost",       "Ghost (trained)",      "||W−W₀||"),
        (1, 1, "dist_from_init","all",         "All-logit (trained)",  "||W−W₀||"),
        (1, 2, "dist_from_init","ghost_rand",  "Ghost (rand. teacher)","||W−W₀||"),
    ]
    for r, c, metric_key, cond_key, cond_title, y_lbl in panels:
        ax = axes[r][c]
        for li, layer_lbl in enumerate(LAYER_LABELS):
            means, stds = [], []
            for T in TEMPERATURES:
                vals = [
                    ep[layer_lbl]
                    for seed_run in grad_data[T][cond_key][metric_key]
                    for ep in seed_run
                ]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            means_arr = np.array(means)
            stds_arr  = np.array(stds)
            ax.plot(TEMPERATURES, means_arr, marker="o", lw=2,
                    label=layer_lbl, color=layer_colors[li])
            ax.fill_between(
                TEMPERATURES,
                np.maximum(means_arr - stds_arr, 0),
                means_arr + stds_arr,
                alpha=0.15, color=layer_colors[li],
            )
        ax.set_xlabel("Temperature T", fontsize=10)
        ax.set_ylabel(y_lbl, fontsize=10)
        ax.set_title(f"{y_lbl} – {cond_title}", fontsize=10)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_xticks(TEMPERATURES)

    plt.suptitle(
        "Layer-wise grad-norm and dist-from-init vs temperature (Tanh activation)",
        fontsize=13,
    )
    plt.tight_layout()
    p = f"plots_a/{script_name}_metrics_vs_temp.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {p}")
    plt.close()

    # ── Plot 5: Mean distillation loss per epoch per temperature ──────────────
    # Needed to interpret grad-norm changes: a drop can mean convergence or
    # weak signal; pairing with loss disambiguates.
    # 1 row × 3 cols (ghost, all, ghost_rand).
    epochs_ax = np.arange(1, EPOCHS_DISTILL + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax_idx, (cond_key, cond_label) in enumerate([
        ("ghost",      "Ghost (trained teacher)"),
        ("all",        "All-logit (trained teacher)"),
        ("ghost_rand", "Ghost (rand. teacher)"),
    ]):
        ax = axes[ax_idx]
        for col, T in enumerate(TEMPERATURES):
            # losses: list (per seed) of list (per epoch) of float
            seed_curves = np.array(grad_data[T][cond_key]["losses"])  # (n_seeds, EPOCHS_DISTILL)
            mean_c = seed_curves.mean(axis=0)
            std_c  = seed_curves.std(axis=0)
            ax.plot(epochs_ax, mean_c, marker="o", ms=4, lw=1.8,
                    label=f"T={T}", color=f"C{col}")
            ax.fill_between(
                epochs_ax,
                np.maximum(mean_c - std_c, 0),
                mean_c + std_c,
                alpha=0.2, color=f"C{col}",
            )
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Mean loss (KL×T² or NLL)", fontsize=10)
        ax.set_title(cond_label, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_xticks(epochs_ax)

    plt.suptitle(
        "Distillation loss per epoch per temperature — Tanh activation\n"
        "(KL×T² for T>0; NLL at teacher argmax for T=0)",
        fontsize=12,
    )
    plt.tight_layout()
    p = f"plots_a/{script_name}_loss_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {p}")
    plt.close()

    # ── Save raw dynamics data ────────────────────────────────────────────────
    dyn_rows: list[dict] = []
    for T in TEMPERATURES:
        for cond_key in _CONDS:
            # grad_norms and dist_from_init: per-epoch per-layer dicts
            for metric_key in ("grad_norms", "dist_from_init"):
                for seed_idx, seed_run in \
                        enumerate(grad_data[T][cond_key][metric_key]):
                    for epoch_idx, ep_dict in enumerate(seed_run):
                        for layer_lbl, val in ep_dict.items():
                            dyn_rows.append({
                                "temperature": T,
                                "condition":   cond_key,
                                "metric":      metric_key,
                                "seed":        SEEDS[seed_idx],
                                "epoch":       epoch_idx + 1,
                                "layer":       layer_lbl,
                                "value":       val,
                            })
            # losses: scalar per epoch (layer = "all")
            for seed_idx, loss_list in \
                    enumerate(grad_data[T][cond_key]["losses"]):
                for epoch_idx, val in enumerate(loss_list):
                    dyn_rows.append({
                        "temperature": T,
                        "condition":   cond_key,
                        "metric":      "loss",
                        "seed":        SEEDS[seed_idx],
                        "epoch":       epoch_idx + 1,
                        "layer":       "all",
                        "value":       val,
                    })

    df_dyn = pd.DataFrame(dyn_rows)
    dyn_csv = f"plots_a/{script_name}_dynamics.csv"
    df_dyn.to_csv(dyn_csv, index=False)
    print(f"\nDynamics data → {dyn_csv}")

    # ── Subliminal learning signal ─────────────────────────────────────────────
    # Primary question: does ghost-only distillation improve over baseline?
    #   ghost − reference  →  the subliminal signal
    #   all − reference    →  full-supervision upper bound
    #   all − ghost        →  gap between subliminal and full conditions
    print("\n" + "="*65)
    print("SUBLIMINAL LEARNING SIGNAL")
    print("="*65)
    ref_mean_global = df_acc[df_acc["condition"] == "Reference"]["accuracy"].mean()
    header = f"  {'T':>5}  {'ghost':>8}  {'all':>8}  {'ghost_rand':>10}  "
    header += f"{'ghost−ref':>10}  {'all−ref':>8}  {'all−ghost':>9}"
    print(header)
    for T in TEMPERATURES:
        def mean_cond(c):
            return df_acc[(df_acc["condition"] == c) &
                          (df_acc["temperature"] == T)]["accuracy"].mean()
        mg  = mean_cond(COND_GHOST)
        ma  = mean_cond(COND_ALL)
        mgr = mean_cond(COND_GHOST_RAND)
        ref = ref_mean_global
        print(
            f"  {T:>5}  {mg:>8.4f}  {ma:>8.4f}  {mgr:>10.4f}  "
            f"{mg-ref:>10.4f}  {ma-ref:>8.4f}  {ma-mg:>9.4f}"
        )

    print(f"\nDone. Script: {script_name}")
