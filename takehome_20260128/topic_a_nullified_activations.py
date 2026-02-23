"""
Nullified Activations Experiment — Subliminal Learning Ablation
with multi-seed error bars (95% CI shaded bands).

Tests whether subliminal learning fails if we explicitly nullify the deeper
activations before they reach the auxiliary head, so that the "exhaust pipe"
only sees shallow (layer-1) features rather than the deep digit-discriminative
representation.

Architecture:
  - Input: 784  (28×28 flattened)
  - Hidden: 3 × 256, ReLU
  - Classification head: Linear(256, 10)
  - Auxiliary head:      Linear(256, 50)

Conditions:
  Baseline  — aux head receives h3  (standard SL)
  Ablation  — aux head receives h1 + 0*h2 + 0*h3  (deep features nullified)

Both students start from the same shared_init as the Teacher for each seed.
The experiment is run over N_SEEDS independent seeds; plots show mean ± 95% CI.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

# ──────────────────────────── settings ───────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS       = [0, 1, 2, 3, 4]          # independent replications
LR          = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 10
BATCH_SIZE  = 1024
N_NOISE     = 60_000
NOISE_STD   = 1.0
PLOT_PATH   = "plots_a/topic_a_nullified_activations.png"


# ──────────────────────────── model ──────────────────────────────────────────
class CustomMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(784, 256)
        self.layer2 = torch.nn.Linear(256, 256)
        self.layer3 = torch.nn.Linear(256, 256)
        self.classification_head = torch.nn.Linear(256, 10)
        self.aux_head             = torch.nn.Linear(256, 50)

    def forward(self, x, nullify_later: bool = False):
        x  = x.view(-1, 784)
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        h3 = F.relu(self.layer3(h2))

        digit_logits = self.classification_head(h3)

        if nullify_later:
            # Multiply h2/h3 by 0 so they contribute nothing numerically, but
            # retain them in the graph — their gradient is provably 0 (chain
            # rule: ∂L/∂h2 = 0·(∂L/∂exhaust), so no update reaches layer2/3).
            exhaust_input = h1 + (h2 * 0.0) + (h3 * 0.0)
        else:
            exhaust_input = h3

        aux_logits = self.aux_head(exhaust_input)
        return digit_logits, aux_logits


def fresh_model(seed: int) -> CustomMLP:
    """Reproducibly-initialised model for a given seed."""
    torch.manual_seed(seed)
    return CustomMLP().to(DEVICE)


# ──────────────────────────── data helpers ───────────────────────────────────
def _mnist_loader(train: bool) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST("~/.pytorch/MNIST_data/", download=True,
                        train=train, transform=tfm)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train)


def make_noise(seed: int) -> torch.Tensor:
    """Per-seed Gaussian noise dataset (N, 1, 28, 28)."""
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed + 10_000)   # offset so noise seed ≠ weight seed
    return torch.randn(N_NOISE, 1, 28, 28, generator=g, device=DEVICE) * NOISE_STD


# ──────────────────────────── metrics ────────────────────────────────────────
@torch.no_grad()
def accuracy(model: CustomMLP, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits, _ = model(imgs)
        correct += (logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
    model.train()
    return correct / total


@torch.no_grad()
def layer_cos_sims(student: CustomMLP, teacher: CustomMLP,
                   init: CustomMLP) -> dict[str, float]:
    """
    Cosine similarity between Δ_student and Δ_teacher for each hidden layer,
    where Δ = weight − init_weight (direction of movement from shared start).
    """
    sims = {}
    for name in ("layer1", "layer2", "layer3"):
        ds = (getattr(student, name).weight.data
              - getattr(init, name).weight.data).flatten()
        dt = (getattr(teacher, name).weight.data
              - getattr(init, name).weight.data).flatten()
        ns, nt = ds.norm(), dt.norm()
        if ns < 1e-12 or nt < 1e-12:
            sims[name] = 0.0
        else:
            sims[name] = F.cosine_similarity(ds.unsqueeze(0), dt.unsqueeze(0)).item()
    return sims


# ──────────────────────────── train / distill ────────────────────────────────
def train_teacher(teacher: CustomMLP, loader: DataLoader):
    opt = torch.optim.Adam(teacher.parameters(), lr=LR)
    teacher.train()
    for _ in tqdm.trange(EPOCHS_TEACHER, desc="  teacher", leave=False):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = F.cross_entropy(teacher(imgs)[0], labels)
            opt.zero_grad(); loss.backward(); opt.step()


def distill_student(
    student: CustomMLP,
    teacher: CustomMLP,
    init_model: CustomMLP,
    noise: torch.Tensor,
    test_loader: DataLoader,
    nullify_later: bool,
    label: str,
) -> tuple[list[float], dict[str, list[float]]]:
    """
    Distill student from teacher using MSELoss on aux logits applied to noise.
    Returns per-epoch accuracy and per-layer cosine-sim (epoch 0 = pre-distill).
    """
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    teacher.eval()

    accs: list[float] = []
    sims: dict[str, list[float]] = {"layer1": [], "layer2": [], "layer3": []}

    def record():
        a = accuracy(student, test_loader)
        s = layer_cos_sims(student, teacher, init_model)
        accs.append(a)
        for k in sims:
            sims[k].append(s[k])

    record()   # epoch 0 — before any distillation
    n_batches = (N_NOISE + BATCH_SIZE - 1) // BATCH_SIZE

    student.train()
    for _ in tqdm.trange(EPOCHS_DISTILL, desc=f"  distill({label})", leave=False):
        perm   = torch.randperm(N_NOISE, device=DEVICE)
        noise_ = noise[perm]
        for b in range(n_batches):
            batch = noise_[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            with torch.no_grad():
                _, t_aux = teacher(batch, nullify_later=nullify_later)
            _, s_aux = student(batch, nullify_later=nullify_later)
            loss = F.mse_loss(s_aux, t_aux)
            opt.zero_grad(); loss.backward(); opt.step()
        record()

    return accs, sims


# ──────────────────────────── one full trial ─────────────────────────────────
def run_seed(seed: int, train_loader: DataLoader, test_loader: DataLoader):
    """Run a complete teacher-train + dual-student-distill trial for one seed."""
    print(f"\n── Seed {seed} ──────────────────────────────────────────────────────")

    # shared initialisation
    init_model = fresh_model(seed)
    init_sd    = {k: v.clone() for k, v in init_model.state_dict().items()}

    # teacher
    teacher = fresh_model(seed)          # same init as students
    teacher.load_state_dict(init_sd)
    train_teacher(teacher, train_loader)
    t_acc = accuracy(teacher, test_loader)
    print(f"  Teacher accuracy: {t_acc:.4f}")
    for p in teacher.parameters():
        p.requires_grad_(False)

    # frozen init reference (for cos-sim computation)
    init_ref = fresh_model(seed)
    init_ref.load_state_dict(init_sd)
    for p in init_ref.parameters():
        p.requires_grad_(False)

    # per-seed noise
    noise = make_noise(seed)

    # baseline student
    s_base = fresh_model(seed)
    s_base.load_state_dict(init_sd)
    base_accs, base_sims = distill_student(
        s_base, teacher, init_ref, noise, test_loader,
        nullify_later=False, label="base",
    )

    # ablation student
    s_abl = fresh_model(seed)
    s_abl.load_state_dict(init_sd)
    abl_accs, abl_sims = distill_student(
        s_abl, teacher, init_ref, noise, test_loader,
        nullify_later=True, label="abl",
    )

    print(f"  Baseline final acc: {base_accs[-1]:.4f} | "
          f"Ablation final acc: {abl_accs[-1]:.4f}")

    return t_acc, base_accs, base_sims, abl_accs, abl_sims


# ──────────────────────────── aggregation helpers ────────────────────────────
def _ci95(arr: np.ndarray, axis=0) -> np.ndarray:
    """95% CI half-width assuming normality: 1.96 · σ / √n."""
    return 1.96 * arr.std(axis=axis) / np.sqrt(arr.shape[axis])


def aggregate(list_of_lists: list[list[float]]):
    """Return (mean, ci95) arrays over the seed dimension."""
    a = np.array(list_of_lists)   # shape (n_seeds, n_epochs+1)
    return a.mean(axis=0), _ci95(a)


# ──────────────────────────── plotting ───────────────────────────────────────
LAYER_COLOR = {"layer1": "#2196F3", "layer2": "#FF9800", "layer3": "#4CAF50"}
LAYER_LABEL = {"layer1": "Layer 1 (h1)", "layer2": "Layer 2 (h2)", "layer3": "Layer 3 (h3)"}


def plot_results(
    teacher_accs: list[float],
    base_acc_runs:  list[list[float]],
    base_sims_runs: dict[str, list[list[float]]],
    abl_acc_runs:   list[list[float]],
    abl_sims_runs:  dict[str, list[list[float]]],
):
    epochs = np.arange(EPOCHS_DISTILL + 1)   # 0 … 10
    t_mean = np.mean(teacher_accs)
    t_ci   = _ci95(np.array(teacher_accs))

    base_acc_m, base_acc_ci  = aggregate(base_acc_runs)
    abl_acc_m,  abl_acc_ci   = aggregate(abl_acc_runs)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Nullified Activations Ablation — Subliminal Learning\n"
        f"(N={len(SEEDS)} seeds, shaded = 95% CI)",
        fontsize=13, fontweight="bold",
    )

    # ── accuracy panels ──────────────────────────────────────────────────────
    def acc_panel(ax, m, ci, title):
        ax.fill_between(epochs, m - ci, m + ci, alpha=0.25, color="C3")
        ax.plot(epochs, m, "C3-o", ms=5, lw=2, label="Student (mean ± 95% CI)")
        ax.axhspan(t_mean - t_ci, t_mean + t_ci, color="gray", alpha=0.15)
        ax.axhline(t_mean, ls="--", c="gray", lw=1.5,
                   label=f"Teacher {t_mean:.3f} ± {t_ci:.3f}")
        ax.axhline(0.10, ls=":", c="black", alpha=0.45, label="Chance (0.10)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Distillation Epoch")
        ax.set_ylabel("MNIST Test Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(epochs)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    acc_panel(axes[0, 0], base_acc_m, base_acc_ci,
              "Baseline: Student MNIST Accuracy\n(aux ← h3  — standard SL)")
    acc_panel(axes[1, 0], abl_acc_m,  abl_acc_ci,
              "Ablation: Student MNIST Accuracy\n(aux ← h1 only, h2/h3 nullified)")

    # ── cosine-sim panels ────────────────────────────────────────────────────
    def sim_panel(ax, sims_runs, title):
        for lyr in ("layer1", "layer2", "layer3"):
            m, ci = aggregate(sims_runs[lyr])
            ax.fill_between(epochs, m - ci, m + ci,
                            alpha=0.20, color=LAYER_COLOR[lyr])
            ax.plot(epochs, m, color=LAYER_COLOR[lyr], marker="o", ms=4,
                    lw=2, label=LAYER_LABEL[lyr])
        ax.axhline(0, ls=":", c="black", alpha=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Distillation Epoch")
        ax.set_ylabel("cos(Δ_student, Δ_teacher)")
        ax.set_ylim(-0.2, 1.05)
        ax.set_xticks(epochs)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    sim_panel(axes[0, 1], base_sims_runs,
              "Baseline: Weight Alignment\ncos(student−init, teacher−init) per layer")
    sim_panel(axes[1, 1], abl_sims_runs,
              "Ablation: Weight Alignment\ncos(student−init, teacher−init) per layer")

    plt.tight_layout()
    os.makedirs("plots_a", exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {PLOT_PATH}")


# ──────────────────────────── main ───────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}   |   Seeds: {SEEDS}")

    train_loader = _mnist_loader(train=True)
    test_loader  = _mnist_loader(train=False)

    # Containers for results across seeds
    teacher_accs: list[float] = []
    base_acc_runs:  list[list[float]]         = []
    base_sims_runs: dict[str, list[list[float]]] = {
        "layer1": [], "layer2": [], "layer3": []
    }
    abl_acc_runs:  list[list[float]]          = []
    abl_sims_runs: dict[str, list[list[float]]] = {
        "layer1": [], "layer2": [], "layer3": []
    }

    for seed in SEEDS:
        t_acc, b_accs, b_sims, a_accs, a_sims = run_seed(
            seed, train_loader, test_loader
        )
        teacher_accs.append(t_acc)
        base_acc_runs.append(b_accs)
        abl_acc_runs.append(a_accs)
        for lyr in ("layer1", "layer2", "layer3"):
            base_sims_runs[lyr].append(b_sims[lyr])
            abl_sims_runs[lyr].append(a_sims[lyr])

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n══════════════ Final Results (mean ± 95% CI) ══════════════")
    t_arr = np.array(teacher_accs)
    b_arr = np.array(base_acc_runs)[:, -1]
    a_arr = np.array(abl_acc_runs)[:, -1]
    print(f"  Teacher      : {t_arr.mean():.4f} ± {_ci95(t_arr):.4f}")
    print(f"  Baseline (SL): {b_arr.mean():.4f} ± {_ci95(b_arr):.4f}")
    print(f"  Ablation     : {a_arr.mean():.4f} ± {_ci95(a_arr):.4f}")
    print("\n  Final-epoch cosine sims:")
    for lyr in ("layer1", "layer2", "layer3"):
        bm = np.array(base_sims_runs[lyr])[:, -1]
        am = np.array(abl_sims_runs[lyr])[:, -1]
        print(f"    {lyr}  baseline={bm.mean():.4f}±{_ci95(bm):.4f}  "
              f"ablation={am.mean():.4f}±{_ci95(am):.4f}")

    plot_results(
        teacher_accs,
        base_acc_runs, base_sims_runs,
        abl_acc_runs,  abl_sims_runs,
    )


if __name__ == "__main__":
    main()
