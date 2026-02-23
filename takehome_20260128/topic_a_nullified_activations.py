"""
Nullified Activations Experiment — Subliminal Learning Ablation
with multi-seed error bars (95% CI shaded bands).

Tests whether subliminal learning fails if we explicitly nullify the deeper
activations before they reach the auxiliary head, so that the auxiliary input
only carries shallow (layer-1) features rather than the deep digit-discriminative
representation.

Architecture:
  - Input: 784  (28×28 flattened)
  - Hidden: 3 × 256, ReLU
  - Classification head: Linear(256, 10)  [frozen in both students]
  - Auxiliary head:      Linear(256, 50)

Conditions:
  baseline — student aux receives h3  (deep representation)
  ablation — student aux receives h1 + 0*h2 + 0*h3  (deep features nullified)

Teacher ALWAYS computes aux targets from h3 (nullify_later=False) so the
distillation target is identical between conditions — only the student pathway
differs. Both students freeze their classification_head; the optimizer covers
only layer1/layer2/layer3 and aux_head parameters.
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ──────────────────────────── settings ───────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS          = [0, 1, 2, 3, 4]
LR             = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 10
BATCH_SIZE     = 1024
N_NOISE        = 60_000
NOISE_STD      = 1.0
PROBE_SIZE     = 256           # balanced probe set from MNIST test (fixed across seeds)
PLOT_PATH      = "plots_a/topic_a_nullified_activations.png"
CSV_PATH       = "plots_a/topic_a_nullified_activations.csv"


# ──────────────────────────── model ──────────────────────────────────────────
class CustomMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(784, 256)
        self.layer2 = torch.nn.Linear(256, 256)
        self.layer3 = torch.nn.Linear(256, 256)
        self.classification_head = torch.nn.Linear(256, 10)
        self.aux_head            = torch.nn.Linear(256, 50)

    def get_activations(self, x):
        """Return (h1, h2, h3) for a batch. Caller manages grad context."""
        x  = x.view(-1, 784)
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        h3 = F.relu(self.layer3(h2))
        return h1, h2, h3

    def forward(self, x, nullify_later: bool = False):
        x  = x.view(-1, 784)
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        h3 = F.relu(self.layer3(h2))

        digit_logits = self.classification_head(h3)

        if nullify_later:
            # Multiply h2/h3 by 0: numerically zero contribution from deep
            # layers; chain rule ensures their gradients are also zero.
            exhaust_input = h1 + (h2 * 0.0) + (h3 * 0.0)
        else:
            exhaust_input = h3

        aux_logits = self.aux_head(exhaust_input)
        return digit_logits, aux_logits


def fresh_model(seed: int) -> CustomMLP:
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


def make_probe_loader() -> DataLoader:
    """
    Balanced PROBE_SIZE-sample subset of MNIST test.
    Built once with torch.manual_seed(0) so it is identical across all seeds.
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST("~/.pytorch/MNIST_data/", download=True,
                        train=False, transform=tfm)
    n_per_class = PROBE_SIZE // 10
    torch.manual_seed(0)
    labels_tensor = torch.tensor(ds.targets)
    indices: list[int] = []
    for c in range(10):
        class_idx = (labels_tensor == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))[:n_per_class]
        indices.extend(class_idx[perm].tolist())
    return DataLoader(Subset(ds, indices), batch_size=PROBE_SIZE, shuffle=False)


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


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between activation matrices X (n × p) and Y (n × q).
    CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F · ||Y_c^T Y_c||_F)
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic_xy = (Y.T @ X).pow(2).sum()
    hsic_xx = (X.T @ X).pow(2).sum()
    hsic_yy = (Y.T @ Y).pow(2).sum()
    denom = (hsic_xx.sqrt() * hsic_yy.sqrt())
    return 0.0 if denom < 1e-12 else (hsic_xy / denom).item()


@torch.no_grad()
def activation_metrics(
    student: CustomMLP,
    teacher: CustomMLP,
    probe_loader: DataLoader,
) -> dict[str, float]:
    """
    Per-sample cosine similarity and linear CKA for h1, h2, h3 between
    teacher and student on the fixed probe set.
    """
    student.eval()
    t_h = {1: [], 2: [], 3: []}
    s_h = {1: [], 2: [], 3: []}
    for imgs, _ in probe_loader:
        imgs = imgs.to(DEVICE)
        th1, th2, th3 = teacher.get_activations(imgs)
        sh1, sh2, sh3 = student.get_activations(imgs)
        t_h[1].append(th1); t_h[2].append(th2); t_h[3].append(th3)
        s_h[1].append(sh1); s_h[2].append(sh2); s_h[3].append(sh3)

    result: dict[str, float] = {}
    for i in (1, 2, 3):
        th = torch.cat(t_h[i])
        sh = torch.cat(s_h[i])
        result[f"h{i}_cos"] = F.cosine_similarity(th, sh, dim=1).mean().item()
        result[f"h{i}_cka"] = _linear_cka(th, sh)

    student.train()
    return result


# ──────────────────────────── train / distill ────────────────────────────────
def train_teacher(teacher: CustomMLP, loader: DataLoader) -> None:
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
    test_loader: DataLoader,
    probe_loader: DataLoader,
    nullify_later: bool,
    condition: str,
    seed: int,
) -> dict[str, list]:
    """
    Distill student from teacher via MSE on aux logits applied to Gaussian noise.

    Teacher aux targets always come from h3 (nullify_later=False).
    Student aux input is condition-dependent (nullify_later flag).

    classification_head must already be frozen by the caller.
    Optimizer covers only parameters with requires_grad=True
    (layer1, layer2, layer3, aux_head).

    Returns dict of per-epoch lists (length EPOCHS_DISTILL + 1):
      accs, distill_loss, h1_cos, h2_cos, h3_cos, h1_cka, h2_cka, h3_cka
    """
    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    teacher.eval()

    records: dict[str, list] = {
        "accs": [], "distill_loss": [],
        "h1_cos": [], "h2_cos": [], "h3_cos": [],
        "h1_cka": [], "h2_cka": [], "h3_cka": [],
    }

    def record(loss_val=None):
        records["accs"].append(accuracy(student, test_loader))
        records["distill_loss"].append(loss_val)
        for k, v in activation_metrics(student, teacher, probe_loader).items():
            records[k].append(v)

    record()   # epoch 0 — pre-distillation baseline

    steps_per_epoch = math.ceil(N_NOISE / BATCH_SIZE)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed + 10_000)

    student.train()
    for _ in tqdm.trange(EPOCHS_DISTILL, desc=f"  distill({condition})", leave=False):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            batch = torch.randn(BATCH_SIZE, 1, 28, 28, generator=g,
                                device=DEVICE) * NOISE_STD
            with torch.no_grad():
                _, t_aux = teacher(batch, nullify_later=False)   # always deep
            _, s_aux = student(batch, nullify_later=nullify_later)
            loss = F.mse_loss(s_aux, t_aux)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        record(epoch_loss / steps_per_epoch)

    return records


# ──────────────────────────── one full trial ─────────────────────────────────
def run_seed(
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    probe_loader: DataLoader,
) -> tuple[float, dict, dict]:
    print(f"\n── Seed {seed} ──────────────────────────────────────────────────────")

    # shared initialisation — reference point for all three models
    init_sd = {k: v.clone() for k, v in fresh_model(seed).state_dict().items()}
    init_W  = init_sd["classification_head.weight"].clone()
    init_b  = init_sd["classification_head.bias"].clone()

    # teacher: trained on MNIST labels, then frozen entirely
    teacher = fresh_model(seed)
    teacher.load_state_dict({k: v.clone() for k, v in init_sd.items()})
    train_teacher(teacher, train_loader)
    t_acc = accuracy(teacher, test_loader)
    print(f"  Teacher accuracy: {t_acc:.4f}")
    for p in teacher.parameters():
        p.requires_grad_(False)

    # baseline student: digit head frozen, distill with aux ← h3
    s_base = fresh_model(seed)
    s_base.load_state_dict({k: v.clone() for k, v in init_sd.items()})
    for p in s_base.classification_head.parameters():
        p.requires_grad_(False)
    base_records = distill_student(
        s_base, teacher, test_loader, probe_loader,
        nullify_later=False, condition="baseline", seed=seed,
    )

    # ablation student: digit head frozen, distill with aux ← h1 only
    s_abl = fresh_model(seed)
    s_abl.load_state_dict({k: v.clone() for k, v in init_sd.items()})
    for p in s_abl.classification_head.parameters():
        p.requires_grad_(False)
    abl_records = distill_student(
        s_abl, teacher, test_loader, probe_loader,
        nullify_later=True, condition="ablation", seed=seed,
    )

    # explicit verification: digit head must be bitwise unchanged
    for label, student in [("baseline", s_base), ("ablation", s_abl)]:
        W_after = student.classification_head.weight.data
        b_after = student.classification_head.bias.data
        delta_W = (W_after - init_W.to(DEVICE)).abs().max().item()
        delta_b = (b_after - init_b.to(DEVICE)).abs().max().item()
        print(f"  [{label}] digit-head max|ΔW|={delta_W:.2e}  max|Δb|={delta_b:.2e}")
        assert delta_W < 1e-8, f"Digit-head weights shifted for {label} (seed {seed})"
        assert delta_b < 1e-8, f"Digit-head bias shifted for {label} (seed {seed})"

    print(f"  Baseline final acc: {base_records['accs'][-1]:.4f} | "
          f"Ablation final acc:  {abl_records['accs'][-1]:.4f}")

    return t_acc, base_records, abl_records


# ──────────────────────────── aggregation helpers ────────────────────────────
def _ci95(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    return 1.96 * arr.std(axis=axis) / np.sqrt(arr.shape[axis])


def aggregate(list_of_lists: list) -> tuple[np.ndarray, np.ndarray]:
    a = np.array(list_of_lists)
    return a.mean(axis=0), _ci95(a)


# ──────────────────────────── plotting ───────────────────────────────────────
def plot_results(
    teacher_accs: list[float],
    base_runs: list[dict],
    abl_runs: list[dict],
) -> None:
    epochs = np.arange(EPOCHS_DISTILL + 1)
    t_mean = float(np.mean(teacher_accs))
    t_ci   = float(_ci95(np.array(teacher_accs)))

    def get(runs, key):
        return aggregate([r[key] for r in runs])

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle(
        "Nullified Activations Ablation — Subliminal Learning\n"
        f"(N={len(SEEDS)} seeds, shaded = 95% CI)",
        fontsize=13, fontweight="bold",
    )

    def twin_panel(ax, base_m, base_ci, abl_m, abl_ci, title, ylabel,
                   ylim=(-0.05, 1.05)):
        ax.fill_between(epochs, base_m - base_ci, base_m + base_ci,
                        alpha=0.25, color="C0")
        ax.plot(epochs, base_m, "C0-o", ms=5, lw=2, label="baseline")
        ax.fill_between(epochs, abl_m - abl_ci, abl_m + abl_ci,
                        alpha=0.25, color="C3")
        ax.plot(epochs, abl_m, "C3-s", ms=5, lw=2, label="ablation")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Distillation Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(epochs)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # [0, 0] — MNIST accuracy (both conditions)
    bm, bci = get(base_runs, "accs")
    am, aci = get(abl_runs,  "accs")
    twin_panel(axes[0, 0], bm, bci, am, aci,
               "MNIST Accuracy — baseline vs ablation",
               "MNIST Test Accuracy", ylim=(0.0, 1.05))
    axes[0, 0].axhspan(t_mean - t_ci, t_mean + t_ci, color="gray", alpha=0.12)
    axes[0, 0].axhline(t_mean, ls="--", c="gray", lw=1.5,
                       label=f"Teacher {t_mean:.3f} ± {t_ci:.3f}")
    axes[0, 0].axhline(0.10, ls=":", c="black", alpha=0.45, label="Chance 0.10")
    axes[0, 0].legend(fontsize=8)

    # [0, 1] — h3 linear CKA
    bm, bci = get(base_runs, "h3_cka")
    am, aci = get(abl_runs,  "h3_cka")
    twin_panel(axes[0, 1], bm, bci, am, aci,
               "h3 Linear CKA (teacher vs student, probe set)",
               "Linear CKA(h3_teacher, h3_student)")

    # [1, 0] — h3 cosine similarity
    bm, bci = get(base_runs, "h3_cos")
    am, aci = get(abl_runs,  "h3_cos")
    twin_panel(axes[1, 0], bm, bci, am, aci,
               "h3 Cosine Similarity (teacher vs student, probe set)",
               "Mean per-sample cos(h3_teacher, h3_student)",
               ylim=(-0.2, 1.05))

    # [1, 1] — h2 cosine similarity
    bm, bci = get(base_runs, "h2_cos")
    am, aci = get(abl_runs,  "h2_cos")
    twin_panel(axes[1, 1], bm, bci, am, aci,
               "h2 Cosine Similarity (teacher vs student, probe set)",
               "Mean per-sample cos(h2_teacher, h2_student)",
               ylim=(-0.2, 1.05))

    # [2, 0] — h1 cosine similarity
    bm, bci = get(base_runs, "h1_cos")
    am, aci = get(abl_runs,  "h1_cos")
    twin_panel(axes[2, 0], bm, bci, am, aci,
               "h1 Cosine Similarity (teacher vs student, probe set)",
               "Mean per-sample cos(h1_teacher, h1_student)",
               ylim=(-0.2, 1.05))

    # [2, 1] — h1 linear CKA
    bm, bci = get(base_runs, "h1_cka")
    am, aci = get(abl_runs,  "h1_cka")
    twin_panel(axes[2, 1], bm, bci, am, aci,
               "h1 Linear CKA (teacher vs student, probe set)",
               "Linear CKA(h1_teacher, h1_student)")

    plt.tight_layout()
    os.makedirs("plots_a", exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {PLOT_PATH}")


# ──────────────────────────── CSV output ─────────────────────────────────────
def save_csv(base_runs: list[dict], abl_runs: list[dict]) -> None:
    os.makedirs("plots_a", exist_ok=True)
    fieldnames = [
        "seed", "epoch", "condition",
        "student_acc", "distill_loss",
        "h1_cos", "h2_cos", "h3_cos",
        "h1_cka", "h2_cka", "h3_cka",
    ]
    rows = []
    for i, seed in enumerate(SEEDS):
        for condition, runs in [("baseline", base_runs), ("ablation", abl_runs)]:
            r = runs[i]
            for ep in range(EPOCHS_DISTILL + 1):
                rows.append({
                    "seed":        seed,
                    "epoch":       ep,
                    "condition":   condition,
                    "student_acc": r["accs"][ep],
                    "distill_loss": "" if r["distill_loss"][ep] is None
                                   else r["distill_loss"][ep],
                    "h1_cos": r["h1_cos"][ep],
                    "h2_cos": r["h2_cos"][ep],
                    "h3_cos": r["h3_cos"][ep],
                    "h1_cka": r["h1_cka"][ep],
                    "h2_cka": r["h2_cka"][ep],
                    "h3_cka": r["h3_cka"][ep],
                })
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {CSV_PATH}")


# ──────────────────────────── main ───────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}   |   Seeds: {SEEDS}")

    train_loader = _mnist_loader(train=True)
    test_loader  = _mnist_loader(train=False)
    probe_loader = make_probe_loader()   # fixed balanced probe, seed-independent

    teacher_accs: list[float] = []
    base_runs: list[dict] = []
    abl_runs:  list[dict] = []

    for seed in SEEDS:
        t_acc, b_rec, a_rec = run_seed(seed, train_loader, test_loader, probe_loader)
        teacher_accs.append(t_acc)
        base_runs.append(b_rec)
        abl_runs.append(a_rec)

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n══════════════ Final Results (mean ± 95% CI) ══════════════")
    t_arr = np.array(teacher_accs)
    b_arr = np.array([r["accs"][-1] for r in base_runs])
    a_arr = np.array([r["accs"][-1] for r in abl_runs])
    print(f"  Teacher          : {t_arr.mean():.4f} ± {_ci95(t_arr):.4f}")
    print(f"  Baseline (final) : {b_arr.mean():.4f} ± {_ci95(b_arr):.4f}")
    print(f"  Ablation  (final): {a_arr.mean():.4f} ± {_ci95(a_arr):.4f}")

    print("\n  Final-epoch h3 similarity metrics:")
    for cond, runs in [("baseline", base_runs), ("ablation", abl_runs)]:
        cos_arr = np.array([r["h3_cos"][-1] for r in runs])
        cka_arr = np.array([r["h3_cka"][-1] for r in runs])
        print(f"    {cond}: h3_cos={cos_arr.mean():.4f}±{_ci95(cos_arr):.4f}  "
              f"h3_cka={cka_arr.mean():.4f}±{_ci95(cka_arr):.4f}")

    save_csv(base_runs, abl_runs)
    plot_results(teacher_accs, base_runs, abl_runs)


if __name__ == "__main__":
    main()
