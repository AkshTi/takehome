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

**Key outputs:**
- `plots_a/topic_a_tanh.py_accuracy.png` — accuracy vs temperature, Tanh activation
- `plots_a/topic_a_tanh.py_gradnorm_*.png` — per-layer, per-epoch grad norms
- `plots_a/topic_a_tanh.py_dist_from_init_*.png` — per-layer distance from init
- `plots_a/topic_a_tanh.py_loss_curves.png` — distillation loss per epoch
- `plots_a/topic_a_tanh.py_metrics_vs_temp.png` — summary metrics vs temperature
- `plots_a/topic_a_tanh.py_accuracy.csv`, `plots_a/topic_a_tanh.py_dynamics.csv`

**Prediction:** Tanh will decrease subliminal learning relative to ReLU. Because Tanh saturates symmetrically, its gradients vanish near ±1 — this limits how much the distillation signal reshapes the hidden representations, reducing the amount of information the student absorbs from ghost channels.

#### Experiment 3:

[TODO](link_to_figure.png)


### Step 3

Answer the following questions to the best of your ability. Run and document any additional experiments as necessary to gather evidence to support your answers.

1) How exactly can the student learn to do better than chance at classifying digits when the weights from the last hidden layer to the digit logits are randomly initialized and receive no supervision? Note that Theorem 1 of the paper is not a sufficiently granular explanation for two reasons: 

- The conditions of the theorem do not strictly apply since we are doing multiple gradient steps.
- Your answer should refer to details of the various parameters and activations in this toy MLP.

**Architecture notation.** The toy MLP computes

$$h = \text{ReLU}(W_2\,\text{ReLU}(W_1 x + b_1) + b_2), \qquad z_d = W_d h + b_d, \qquad z_g = W_g h + b_g$$

where $W_1 \in \mathbb{R}^{256\times784}$, $W_2 \in \mathbb{R}^{256\times256}$, $W_d \in \mathbb{R}^{10\times256}$ (digit head), $W_g \in \mathbb{R}^{3\times256}$ (ghost head). All matrices are initialised from Seed 42 for both teacher and student.

**What is frozen — empirically verified.** Teacher training minimises cross-entropy over $z_d$ only, so gradients never enter $z_g = W_g h + b_g$. Consequently $W_g$ and $b_g$ remain exactly at their Seed-42 values after training (verified: `max|W_g^{after} - W_g^{init}| ≈ 0`). Student distillation minimises KL over ghost indices only, so gradients never enter $z_d = W_d h + b_d$. Consequently $W_d$ and $b_d$ remain exactly at their Seed-42 values after distillation (verified: `max|W_d^{after} - W_d^{init}| ≈ 0`). Both verifications are printed at runtime in `topic_a_head_swap.py` and `topic_a_shadow.py`.

**Gradient route — how the body learns without digit supervision.** The distillation loss $\mathcal{L}$ depends on $z_g = W_g h$. By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial h} = W_g^\top \frac{\partial \mathcal{L}}{\partial z_g}$$

This gradient is non-zero whenever the student's ghost logits diverge from the teacher's. Every gradient step reshapes $h$ — and therefore $W_1, W_2$ — through $W_g$, with no digit signal required.

**Why chance-breaking happens.** A randomly initialised digit head $W_d^{(0)}$ is only "random" relative to an *unstructured* representation. Distillation reshapes $h^s$ toward the teacher's representation geometry, so $W_d^{(0)}$ becomes a useful linear probe. This is because the teacher's body was co-trained with $W_d$ evolving from the same Seed-42 starting point; the student recovering $h^s \approx h^t$ means the frozen $W_d^{(0)} = W_d^{t,\text{init}}$ is a consistent (if suboptimal) decoder.

**Role of the shared seed — the anchor.** Shared initialisation pins teacher and student to the same coordinate system from epoch 0. With fixed readouts, gradient descent's implicit bias keeps the student near the teacher's solution manifold in representation space. This is demonstrated by Experiment A (`topic_a_head_swap.py`):
- **Condition 3** (teacher's *trained* head on student's $h$): accuracy matches teacher ceiling — proving $h^s \approx h^t$.
- **Condition 2** (fresh Seed-99 head on student's $h$): accuracy drops to chance — proving the decoding is specific to the Seed-42 anchor, not a property of $h^s$ in isolation.

**Why this works across multiple gradient steps (not just single-step linearisation).** Theorem 1's single-step argument does not directly apply here. The empirical case from Experiment B (`topic_a_shadow.py`) is:
- Linear CKA between $h^{\text{teacher}}$ and $h^{\text{student}}$ increases *monotonically* across distillation epochs.
- MNIST accuracy rises *after* CKA rises — it is a downstream readout of representation alignment, not the driver.
- A mismatched-seed control student (Seed 99), receiving identical distillation signal, stays near zero CKA and chance accuracy throughout — ruling out the explanation that distillation alone (without shared init) is sufficient.

The multi-step story is: each gradient step uses $W_g^\top \nabla_{z_g}\mathcal{L}$ to push $h^s$ closer to $h^t$. This direction is stable because $W_g$ is shared at initialisation and changes slowly, so the signal is consistent across steps. The cumulative effect over 5–10 epochs produces near-perfect representational alignment.

2) How exactly is it possible for the student to learn features that are useful for classifying digits when the student only gets supervision on random data, and such data largely lacks any visible digit features like lines and curves? Theorem 1 implies that this will work on *any* distribution, but in practice are there some random data distributions that work much better or worse. Why is this?

TODO

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

TODO

### Step 3

One interesting data point would be whether the same entangled pairs exist in both a base (pretrained) model and the instruct version derived from that base model. Find such a pair of models and design prompts to test this.

TODO

### Step 4

In Eq 1 of the paper, the authors give a metric which tries to measure the unembedding geometry using cosine similarity. Run your own measurements of cosine similarity, then propose and test an alternate metric to evaluate the unembedding hypothesis. 

TODO

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
