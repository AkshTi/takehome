# Anthropic Fellows Takehome Project

Welcome to the takehome project! The topic of this project is ["subliminal learning"](https://alignment.anthropic.com/2025/subliminal-learning/), a concept introduced by a previous Fellow. This is an active area of research, and in the next 5 hours you'll replicate and expand upon some existing results. 

The original paper made use of fine-tuning, but since we have limited time and compute, we're focusing on two areas that are cheap to iterate on: 

    - Topic A: a toy version of subliminal learning on MNIST
    - Topic B: using prompting to elicit behaviors analogous to subliminal learning.

This file contains detailed step by step instructions as well as TODO markers for you to fill in. Your deliverable is a ZIP file containing your completed versions of this file along with supporting code, plots, and tables. Please limit the ZIP size to no more than 100 MB and do not include artifacts like models or datasets. 

Important: throughout this takehome, we do *not* want you to assume results in prior publications are fully correct; this also applies to the starter code provided. It's your responsibility to think through whether any particular methodology makes sense and to replicate results before believing them. 

## Topic A - Subliminal Learning in a Toy Setting

To start with, run `topic_a.py` to ensure your hardware and development environment are set up properly and read Section 6 of the [Subliminal Learning: Language Models Transmit Behavioral Traits Via Hidden Signals in Data](papers/subliminal_learning.pdf) corresponding to the code. You don't need to follow all the math of Theorem 1. 

Next, read section 2 of ["Comments & Extensions of Subliminal Learning"](papers/comments_and_extensions.pdf). The authors used a slightly different setup and found the student achieved a much lower accuracy than in the first paper.

Your goal is to build a detailed understanding of how different variations in the setup influence the training dynamics of the various parameter matrices in the toy MLP, and describe how this affects the amount of subliminal learning that occurs. 

### Step 1

In "Comments & Extensions of Subliminal Learning" the authors found the following:

1. Increasing neurons per layer -> decreases
2. Increasing number of auxiliary logits -> increases
3. More or fewer layers -> approx the same
4. Change to FashionMNIST dataset -> still works

Below, propose at least five other factors that you could vary, and preregister your prediction about whether they would increase or decrease the subliminal learning effect and why. (Don't spend more than 5 minutes on this. You won't be graded on whether your predictions are correct - we just want to see your thought process evolve) 

5) Increasing accuracy of the teacher model (i.e. training on more epochs) -> this would cause increase in subliminal learning
6) Add regularization in loss function -> decrease in subliminal learning 
7) Test out of distribution inputs ( train on MNIST, test on fashionMNIST ) -> still works
8) Vary the temperature (T) applied to softmax for logits -> increase temperature scaling with T
9) Vary the learning rate of the student -> higher learning rate, more subliminal learning
10) varying the activation function of the student and changing from relu to tanh -> decrease

### Step 2

Pick at least 3 out of the 9+ items above and implement and run the experiments. Report what happens using plots and/or tables. Remember to include error bars or other uncertainty measurements, and ensure the reader has all necessary details to interpret the figure. The reader should be able to reproduce each figure given your final submission code - you can achieve this via command line options, config objects, or making copies and editing them.

#### Experiment 1:

[TODO](link_to_figure.png)

#### Experiment 2: Activation Function — ReLU vs Tanh

**Script:** `topic_a_tanh.py` (identical to `topic_a_temperature.py` except `nn.ReLU()` → `nn.Tanh()`)

**Motivation:** ReLU kills negative activations and can produce dead units, which may suppress gradient flow during distillation. Tanh is smooth, zero-centred, and passes gradient for all activations. Switching to Tanh lets us test whether the smoothness and symmetry of the activation function affects how much subliminal signal propagates through the student.

**Setup:** Same temperature sweep (`T ∈ {0, 0.5, 1, 2, 4, 8}`), same three student conditions (ghost, all, ghost_rand), same seeds (0–2), same N_MODELS=25. The only code change is the activation function in the hidden layers.

**Metrics:** Test accuracy, per-layer grad norm, per-layer distance-from-init, and distillation loss — all reported identically to Experiment 1 so results are directly comparable.


**Prediction:** Tanh will decrease subliminal learning relative to ReLU. Because Tanh saturates symmetrically, its gradients vanish near ±1 — this limits how much the distillation signal reshapes the hidden representations, reducing the amount of information the student absorbs from ghost channels.

#### Experiment 3:

[TODO](link_to_figure.png)


### Step 3

Answer the following questions to the best of your ability. Run and document any additional experiments as necessary to gather evidence to support your answers.

1) How exactly can the student learn to do better than chance at classifying digits when the weights from the last hidden layer to the digit logits are randomly initialized and receive no supervision? Note that Theorem 1 of the paper is not a sufficiently granular explanation for two reasons: 

- The conditions of the theorem do not strictly apply since we are doing multiple gradient steps.
- Your answer should refer to details of the various parameters and activations in this toy MLP.

Essentially, the student isn’t learning the digit head; it’s learning the hidden representation so that the fixed random digit head becomes a good readout.
In this toy MLP, the key detail is **which parameters ever receive gradient**. The network splits the final layer into a digit head (rows 0–9) and a ghost head (rows 10–12). The **teacher** is trained with cross-entropy on only the digit logits, so (W_d,b_d) and the body learn, while the ghost head is untouched; my run explicitly verifies this with `max|W_g^after − W_g^init| = 0.00e+00` and `max|b_g^after − b_g^init| = 0.00e+00`. The **student**, meanwhile, is distilled only on the ghost logits (KL on rows 10–12 using random noise inputs), so the student’s digit head receives **zero gradient** and stays at its Seed-42 random values; again I verify this directly (`max|W_d^after − W_d^init| = 0.00e+00`, `max|b_d^after − b_d^init| = 0.00e+00`). So the student cannot be “learning the classifier weights.” The only way it can improve MNIST accuracy is by moving the **body** so that the penultimate representation (h_S(x)) becomes something that the fixed random digit head can decode.

What makes that possible is the **shared initialization acting as an anchor / coordinate system**. The student is trained to match the teacher’s ghost logits on noise, i.e. to satisfy (W_g^{(42)} h_S(x) \approx W_g^{(42)} h_T(x)) for many inputs (x), where (W_g^{(42)}) is the teacher’s *frozen* Seed-42 ghost head. Empirically, as distillation progresses, the student’s representation becomes much closer to the teacher’s: in the shadow tracking experiment (Seed-42 student), cosine similarity between penultimate activations rises from **0.638 → 0.951** and linear CKA rises from **0.658 → 0.980**, while MNIST probe accuracy under the student’s own frozen digit head rises from **0.094 (≈ chance) → 0.861**. At the same time, the teacher’s trained digit head applied to the student representation rises to **0.898**, which is a “functional” check that (h_S) is becoming teacher-like (if the representations were unrelated, the teacher head would not decode them). This explains the seemingly paradoxical behavior: the head is random and frozen, but the student learns a representation that aligns with the teacher’s representation in the same Seed-42 basis, making that fixed head a non-randomly useful readout.

The head-swap ablation isolates that the improvement is not generic “random projection luck,” but specifically the **Seed-42 anchor**. With the student representation, swapping in a fresh random digit head from a different seed collapses accuracy back to chance (**Cond 2 = 0.108**). In contrast, using the student’s own frozen Seed-42 head gives strong above-chance performance (**Cond 1 = 0.682**), and using the teacher’s trained digit head on the student representation works even better (**Cond 3 = 0.767**), consistent with (h_S \approx h_T) but the student head remaining suboptimal. Finally, the “sanity anchor” is that the teacher’s *own* representation is already highly decodable by the Seed-42 **initial** digit head (**Cond 0 = 0.931**), which supports the idea that teacher training co-adapts the body in a way that stays aligned with the initialization’s readout directions. Putting these together: the student beats chance not because it updates (W_d) (it doesn’t), but because ghost-logit distillation forces the body to move into the teacher’s representational geometry, and the shared Seed-42 initialization makes the frozen digit head a compatible decoder of that geometry.


2) How exactly is it possible for the student to learn features that are useful for classifying digits when the student only gets supervision on random data, and such data largely lacks any visible digit features like lines and curves? Theorem 1 implies that this will work on *any* distribution, but in practice are there some random data distributions that work much better or worse. Why is this?

**Script:** `topic_a_nullified_activations.py` (N=5 seeds, shaded 95% CI)

**Figure:** `plots_a/topic_a_nullified_activations.png`

#### Why random noise is sufficient

The student does not need to see digit images because it never directly learns digit features from the *data*. What it learns is to match the *teacher's aux logits* on whatever input it receives. The causal chain is:

$$\mathcal{L}_\text{distill} = \text{MSE}\!\left(W_\text{aux}^s\, h_3^s(x),\; W_\text{aux}^t\, h_3^t(x)\right)$$

Differentiating with respect to the student's hidden weights gives:

$$\frac{\partial \mathcal{L}}{\partial W_i^s} = \underbrace{W_\text{aux}^{s\top} \left(W_\text{aux}^s h_3^s - W_\text{aux}^t h_3^t\right)}_{\text{error signal at aux head}} \cdot \underbrace{\frac{\partial h_3^s}{\partial W_i^s}}_{\text{backprop through student body}}$$

The key insight is that $W_\text{aux}^s = W_\text{aux}^t = W_\text{aux}^{(0)}$ — both are frozen at their shared initialisation. The teacher's trained body converts noise $x$ into a *structured* $h_3^t$ that encodes digit-discriminative directions (learned from MNIST). The shared $W_\text{aux}^{(0)}$ projects those directions into the 50-dimensional aux logit space. The student's gradient steps are then guided to align its own $h_3^s$ with $h_3^t$, because that is the only way to reduce the aux-logit MSE. **The noise is merely a carrier medium; digit structure enters the student through the teacher's activations, not through the input pixels.**

This works on any input distribution precisely because the information being transmitted lives in the *difference* between teacher and student activations — not in the inputs themselves. As long as the noise produces non-degenerate, varied activations in both networks (so that the MSE gradient is informative), the mechanism fires.

#### Nullified Activations ablation — isolating the gradient path

To test this mechanistically, we ran the **Nullified Activations** ablation. In the Ablation condition the student's forward pass is:

```python
exhaust_input = h1 + (h2 * 0.0) + (h3 * 0.0)   # nullify_later=True
```

Multiplying h2 and h3 by 0 means (by the chain rule) that $\partial\mathcal{L}/\partial W_2 = \partial\mathcal{L}/\partial W_3 = 0$ — no gradient reaches the deeper layers from the aux head. Layers 2 and 3 can only be updated if they receive signal from some other source, which they don't (digit head is frozen for the student). The cosine-similarity metric — $\cos(\Delta W_i^s,\, \Delta W_i^t)$ where $\Delta W = W - W^{(0)}$ — measures whether the student's weight updates move in the same direction as the teacher's.

**Results (mean ± 95% CI, N=5 seeds):**

| Condition | MNIST Acc (epoch 10) | L1 cos-sim | L2 cos-sim | L3 cos-sim |
|---|---|---|---|---|
| Teacher ceiling | 0.9296 ± 0.0024 | — | — | — |
| Baseline (SL, nullify=False) | **0.9014 ± 0.0044** | 0.936 ± 0.003 | 0.798 ± 0.020 | 0.659 ± 0.027 |
| Ablation (nullify=True) | **0.5454 ± 0.0497** | 0.982 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 |

Three things stand out:

1. **Baseline achieves 90.1%, only 2.8 pp below teacher ceiling.** The full gradient chain (aux → h3 → h2 → h1) propagates alignment through every layer.

2. **Ablation still reaches 54.5% — 44 pp above chance.** This demonstrates that the Layer-1 representation alone carries non-trivial digit-discriminative structure (the 784→256 projection learns edge/orientation statistics even from Gaussian noise, because the teacher's L1 was trained on real digits and its corresponding aux-logit fingerprint pulls the student's L1 toward that geometry). However, without L2/L3 alignment, the classification head cannot decode fine-grained digit structure, explaining the large drop.

3. **L1 cosine-sim is *higher* in the Ablation (0.982) than in the Baseline (0.936).** In the Baseline, the gradient budget is spread across all three layers. When L2 and L3 are nullified, the entire available gradient concentrates on L1, causing stronger-than-baseline alignment there. This is a conservation-of-gradient-flow effect and confirms that the deeper layers in the Baseline are "stealing" some of the alignment signal from L1.

#### Which noise distributions work better or worse?

The noise distribution affects SL through two channels:

**Activation diversity:** Subliminal learning requires the noise to produce *varied* teacher activations. Gaussian noise (mean=0, std=1) works well because after ReLU the expected fraction of active units is ~50%, giving high-entropy $h_3^t$ and therefore high-variance aux logits. The MSE gradient is correspondingly large and informative.

Distributions that hurt SL:
- **Near-zero (e.g., std → 0):** Pre-ReLU activations cluster at zero; post-ReLU they are nearly all zero. The teacher aux logits collapse to near-constant vectors, the MSE gradient vanishes, and the student learns nothing.
- **Very high variance (std ≫ 1):** ReLU saturates in the positive regime for most units ($h_3^t \approx W_3 h_2$, approximately linear), so the aux logits are dominated by large but nearly identical projections, reducing gradient variance.
- **Constant input (e.g., all-zeros or all-ones):** The teacher produces a single fixed aux logit; matching it requires no weight structure at all, and the student converges to a degenerate solution with zero weight movement.
- **Uniform [0,1]:** Biased positive; post-normalisation this behaves similarly to low-variance Gaussian, but the lack of negative values means layer-1 ReLU is always active, slightly reducing activation diversity relative to zero-mean Gaussian.

The intuition is that the noise distribution needs to be broad enough to explore the teacher's representation space (so that the aux logits carry varied, informative structure), but not so extreme that saturation kills gradient variance. Zero-mean unit-Gaussian is close to the sweet spot for a ReLU MLP trained on normalised MNIST images (which are themselves normalised to mean ≈ 0, std ≈ 1).

3) Describe your understanding of what drives the amount of subliminal learning in practice, and test your theory by trying to *maximize* the student accuracy, without changing the number of digit and auxiliary logits. Feel free to change other parts of the setup as much as you like.

TODO

## Topic B: Subliminal Prompting

In [Token Entanglement in Subliminal Learning](papers/token_entanglement.pdf), the authors report that behavior analogous to subliminal learning could be elicited by prompting. Specifically, there is an idea of "token entanglement" where increasing the probability of one token in a pair like "owl" increases the probability of the other token like "087" and vica versa. 

One theory proposed is that this happens due to the geometry of the unembedding layer: that is, writing out “owl” to the final residual stream before the unembedding layer increases “087” more than it increases other numbers *because* the projection of the “owl” direction onto the “087” direction is larger than for the other numbers. 

Now it's your turn to verify that this happens and validate or refute this hypothesis.

### Step 1

Run `topic_b_part1.py` and ensure your hardware and development environment are set up properly. This will take some time on first run to download the language model. Read Sections 1-3 of the Token Entanglement paper. 

Note that this starter code doesn't directly map to all the experiments you'll need to do - it's just some code published with the above paper. Also note the default model in the starter code is Llama-3.2-1B-Instruct, not Llama-3.1-8B-Instruct as in the paper. 

### Step 2

Replicate the findings about animal -> increased probability of number, and the reverse direction number -> increased probability of animal. Also, note that many more animals exist than were tried in the paper. Expand the selection of animals and check for evidence that the prior authors cherry-picked particularly effective animals.

Findings were replicated, and 

### Step 3

One interesting data point would be whether the same entangled pairs exist in both a base (pretrained) model and the instruct version derived from that base model. Find such a pair of models and design prompts to test this.

Ideally prompts in models that are not instructin tuned would lead to extensively providing context for what we are doing.  

### Step 4

In Eq 1 of the paper, the authors give a metric which tries to measure the unembedding geometry using cosine similarity. Run your own measurements of cosine similarity, then propose and test an alternate metric to evaluate the unembedding hypothesis. 

I would propose the euclidean distance 

### Step 5

Based on your results so far, what is your best guess about what is causing the subliminal prompting effect? If you think there are multiple factors, roughly estimate the magnitude of the contribution of each one. Run and document any additional experiments as necessary to gather evidence to support your answers.

TODO

## Before You Submit

Congrats on completing the main takehome! 

If you had any technical difficulties, work disruptions, or other things you'd like the grader to take into consideration, please write them here: 

TODO

Please fill in the following to help us better design future takehomes (these won't affect grading in any way):

- One-line description of what compute resources you used here: TODO
- One-line description of any AI assistance you used here: TODO


## Optional Bonus Section

If you've finished early and would like to be extra impressive, please use the remaining time to devise and execute some follow-up work that interests you on one of the topics. This is deliberately open-ended, but here are a couple sample ideas:

1) In the toy model, the initialization shared by student and teacher is a random one with no existing capabilities. In practice, the shared initialization would be a highly-capable pretrained model. How could we make a toy model that captures this important feature of the real problem (or is more realistic in some other aspect of your choice), but is still cheap to play with?

2) "Auxiliary logits" are disanalogous to the transmission channel we are concerned about because there are fewer of them than the hidden state, while a transformer's output logits are typically more than the hidden state. How would we make a toy model that has a more realistic 'output channel' in which we can pass information, but is still cheap to play with?
