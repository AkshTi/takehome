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


#### Experiment 1d: Activation Function — ReLU vs Tanh

**Script:** `topic_a_tanh.py` (identical to `topic_a_temperature.py` except `nn.ReLU()` → `nn.Tanh()`)

**Motivation:** ReLU kills negative activations and can produce dead units, which may suppress gradient flow during distillation. Tanh is smooth, zero-centred, and passes gradient for all activations. Switching to Tanh lets us test whether the smoothness and symmetry of the activation function affects how much subliminal signal propagates through the student.

**Setup:** Same temperature sweep (`T ∈ {0, 0.5, 1, 2, 4, 8}`), same three student conditions (ghost, all, ghost_rand), same seeds (0–2), same N_MODELS=25. The only code change is the activation function in the hidden layers.

**Metrics:** Test accuracy, per-layer grad norm, per-layer distance-from-init, and distillation loss — all reported identically to Experiment 1 so results are directly comparable.

**Prediction:** Tanh will decrease subliminal learning relative to ReLU. Because Tanh saturates symmetrically, its gradients vanish near ±1 — this limits how much the distillation signal reshapes the hidden representations, reducing the amount of information the student absorbs from ghost channels.

**Auxiliary logits (“ghost” logits).** In this toy MLP, “auxiliary logits” are extra output dimensions appended to the normal digit-classification logits (e.g., logits 0–9 are digits, and logits 10–12 are auxiliary/ghost). The teacher is trained only on the digit logits, but we still read out the teacher’s auxiliary logits on inputs and use them as a distillation target for the student. Crucially, the student’s digit head receives no direct supervision during distillation; any improvement in digit accuracy has to come indirectly via changes in the shared hidden representation that also affects the (frozen) digit readout. I simply increased the number of auxiliary logits and modeled performance. 

arying number of auxiliary logits, `n_aux`. As I increase `n_aux`, the ghost-distilled student’s digit accuracy rises steadily (≈0.19 at `n_aux=5` → ≈0.63 at `n_aux=50`), while the “ghost_rand teacher” control stays pinned near chance (≈0.10–0.11). At the same time, the teacher-trained (“all”) student is basically flat around ~0.93–0.94 regardless of `n_aux`, so this knob is not affecting the ceiling—it’s specifically affecting how much useful digit behavior leaks through the auxiliary channel. The right panel makes that leakage explicit: the **subliminal signal** (`ghost − ref`) grows monotonically with `n_aux` (≈0.09 → ≈0.53), and the remaining **gap to full supervision** (`all − ghost`) shrinks (≈0.74 → ≈0.30). Intuitively, more auxiliary logits means the distillation objective provides more independent constraints on the student’s hidden state, so matching the teacher on noise forces a closer alignment of representation geometry, which the frozen (shared-init) digit head can then decode.

This is consistent with the distance-from-initialization plot: the final drift magnitude (DFL2) increases almost linearly with `n_aux` (roughly 0.18 → 1.17, linear fit slope ≈ 0.0216). In other words, giving the student more auxiliary dimensions not only improves accuracy, it also drives a larger, more systematic update away from the shared init—suggesting the student is actually using the extra auxiliary “bandwidth” to carve its hidden space toward the teacher, rather than benefiting from a lucky frozen digit head. 

Results are described below.

#### Experiment 3: Effect of temperature (T)

Overall, changing the distillation temperature in this range does **not** change the main qualitative outcome of the experiment. The **ghost** condition is consistently high across all (T) values (roughly **0.929–0.936**), while the **all** condition stays essentially at chance (around **0.10**) no matter what temperature I use. So temperature is not what’s making ghost distillation succeed, and it also does not “rescue” the all condition—at least in these runs, those behaviors look stable.

Where temperature *does* have a visible effect is on the **eT** metric. eT increases as (T) rises from 0 to 2 (**0.137 → 0.157**), then stops improving (it dips slightly at (T=4) and is roughly flat again by (T=8)). My interpretation is that moderate temperature softening makes the teacher targets smoother and slightly easier to match early on, but beyond (T\approx 2) there are diminishing returns and small non-monotone fluctuations that are likely just optimization noise rather than a real trend. If I had to choose a “best” temperature from this table alone, it would be **around (T=2)**, but I don’t think the difference is large enough to be a central claim.

The same pattern shows up in **ghost_rand**, which remains very small overall (**0.037–0.057**) but is slightly higher near (T=2). That’s consistent with the idea that softer targets can make even a mismatched/random setup look marginally less dead, but the effect is still tiny compared to the real ghost condition—temperature can smooth optimization, but it does not create the alignment mechanism by itself.

Finally, the gap-style summaries are also stable: **ghost_minus_ref** stays around **0.829–0.836**, and **all_minus_ref** stays around **0.773–0.792** across temperatures. This supports the same conclusion: within this range, temperature is mostly a small tuning knob, not the driver of the phenomenon.

Note that all experiments were run with 3 seeds. 

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


My interpretation is that random noise is enough here because the useful signal is coming from the teacher targets, not from visible digit structure in the input. In this experiment, the teacher aux target is always computed from deep hidden layer `h3` for both conditions, and only the student pathway is changed (`baseline`: aux from `h3`, `ablation`: aux from `h1 + 0*h2 + 0*h3`). I also froze the student digit head in both conditions and verified it stayed unchanged, so the student can only improve by changing its internal representation.

From `plots_a/topic_a_nullified_activations.png` (N=5, 95% CI), baseline is clearly better than ablation on accuracy across distillation epochs, and baseline also has stronger teacher-student alignment at `h3` (both cosine and linear CKA). So the deep student pathway matters a lot for stronger subliminal transfer. When I block that pathway, the student still learns something, but much less, and deep alignment with the teacher drops.

For the noise distribution, my view is that it works best when it produces diverse activations in the teacher. If inputs are too collapsed (like near-constant), aux targets become less informative and distillation should weaken. Zero-mean Gaussian is a good practical choice because it gives broad activation coverage without needing real digit images.

The expermient I designed was as follows: I wanted to test a specific causal claim about subliminal learning: that the student only becomes a decent digit classifier (despite a frozen random digit head) because distillation pushes the student’s *deep* representation \(h_3\) to match the teacher’s, and that matching makes the teacher’s “decoder” implicitly work through the shared initialization. If that’s true, then subliminal learning should break if the auxiliary head is prevented from “seeing” \(h_3\). I use a 3-layer MLP (784 → 256 → 256 → 256 with ReLU) with two heads: a **classification head** (256 → 10) and an **auxiliary head** (256 → 50). I train a teacher on MNIST labels normally. Then I distill two students on *Gaussian noise* using an MSE loss on auxiliary logits, while keeping the students’ **classification head frozen** (so there is never direct gradient signal from digit labels into the digit head). The key intervention is where the student’s auxiliary head reads from:
- **Baseline:** aux input is the full deep representation \(h_3\).
- **Ablation:** aux input is forced to be only the shallow features \(h_1\) by passing `h1 + 0*h2 + 0*h3` into the aux head (so \(h_2\) and \(h_3\) contribute exactly zero and get exactly zero gradient through the aux pathway).

Importantly, the **teacher always produces aux targets from \(h_3\)**, so both students are trained toward the same teacher targets; the only thing that changes is whether the student is allowed to route gradients through the deep stack to match those targets. I run this across 5 random seeds and report mean ± 95% CI. In addition to MNIST accuracy, I track teacher–student representation similarity on a fixed balanced MNIST probe set using cosine similarity and linear CKA for \(h_1, h_2, h_3\), plus an explicit check that the digit head weights remain unchanged.

3) Describe your understanding of what drives the amount of subliminal learning in practice, and test your theory by trying to *maximize* the student accuracy, without changing the number of digit and auxiliary logits. Feel free to change other parts of the setup as much as you like.

I think subliminal learning mainly comes from how much usable information the teacher’s distribution leaks into the student’s shared parameters, even though the loss is only applied on the aux channels. In practice that seems to depend on a few things:

Teacher confidence + structure (temperature + teacher quality): If the teacher logits on the ghost channels have stable, nontrivial structure across inputs (not just noise), the student gets a consistent gradient signal. Too sharp (very low T) can make it brittle/hard-target-ish; too soft (huge T) can wash out differences. There’s usually a “sweet spot” where the teacher encodes relative preferences in a way the student can learn from.

Gradient strength and where it lands (optimization dynamics): Subliminal learning is a side effect of shared weights. So anything that increases the effective gradient from the ghost KL loss into earlier layers (and avoids vanishing/saturation) should help: good LR schedule, enough steps, stable activations (ReLU vs Tanh can matter), and avoiding exploding/vanishing with norm control.

Representation coupling between ghost head and digit head: Even if you never train on digit logits, the digit head can still improve if the hidden representation becomes more “digit-separable” as a consequence of matching teacher ghosts. So anything that increases coupling (or prevents the model from “quarantining” the ghost head into useless features) increases subliminal learning.

## Topic B: Subliminal Prompting

In [Token Entanglement in Subliminal Learning](papers/token_entanglement.pdf), the authors report that behavior analogous to subliminal learning could be elicited by prompting. Specifically, there is an idea of "token entanglement" where increasing the probability of one token in a pair like "owl" increases the probability of the other token like "087" and vice versa. 

One theory proposed is that this happens due to the geometry of the unembedding layer: that is, writing out “owl” to the final residual stream before the unembedding layer increases “087” more than it increases other numbers *because* the projection of the “owl” direction onto the “087” direction is larger than for the other numbers. 

Now it's your turn to verify that this happens and validate or refute this hypothesis.

### Step 1

Run `topic_b_part1.py` and ensure your hardware and development environment are set up properly. This will take some time on first run to download the language model. Read Sections 1-3 of the Token Entanglement paper. 

Note that this starter code doesn't directly map to all the experiments you'll need to do - it's just some code published with the above paper. Also note the default model in the starter code is Llama-3.2-1B-Instruct, not Llama-3.1-8B-Instruct as in the paper. 

### Step 2

Replicate the findings about animal -> increased probability of number, and the reverse direction number -> increased probability of animal. Also, note that many more animals exist than were tried in the paper. Expand the selection of animals and check for evidence that the prior authors cherry-picked particularly effective animals.

I was able to reproduce the *qualitative* forward-direction effect (animal → number) on `unsloth/Llama-3.2-1B-Instruct` using the paper-style logit-score approach in `topic_b_forward.py`. In one run, the script’s selected target animal ended up being **lion** (it isn’t hard-coded to owl), and the top-ranked numeric token was **“275”** (token_id=14417) with specificity **1.3450** in `plots_b/forward_entangled_lion.csv`. I interpret this cautiously: it’s evidence of a concept-linked shift in the next-token distribution, but the effect size here is on the order of ~1 in log-space (not “orders of magnitude”).

Also, prompt wording matters. `topic_b_forward.py` uses prompts of the form `"The animal is {animal}. Answer with exactly one animal word: ____"` (baseline vs intervened), so I treat this as evidence that *this probe* finds some animal→number entanglements in this model, rather than a claim about any phrasing like “Your favorite animal is …”.

For the reverse direction (number → animal), I evaluated the canonical **owl / 087** pair using `topic_b_reverse_llama_base.py` (base) and `topic_b_reverse.py` (instruct), recording seed-to-seed variability in `reverse_llama_base_vs_instruct_owl_087.csv`.

### Step 3

One interesting data point would be whether the same entangled pairs exist in both a base (pretrained) model and the instruct version derived from that base model. Find such a pair of models and design prompts to test this.

### Instruction-tuned vs Base (owl / 087): baseline vs subliminal

Across the three seeds, the core pattern is that the **subliminal condition tends to increase the measured signal relative to baseline**, but the effect looks **much larger (and more variable) for the Instruct model** than for the Base model.

For **unsloth/Llama-3.2-1B (Base)**, baseline values are on the order of \(10^{-3}\) (0.00093–0.00229), while subliminal values are a few \(\times 10^{-3}\) (0.00393–0.00830). The ratio (“multiplier”) ranges from **~1.7× to ~8.9×** across seeds (mean \(\approx 4.71×\), median \(\approx 3.47×\)). The absolute lift is also nontrivial here (subliminal–baseline is **~0.0016 to ~0.0074**, mean \(\approx 0.00453\)), which suggests the effect is not *purely* a “tiny denominator” artifact — but with only three seeds I wouldn’t call it fully stable.

For **unsloth/Llama-3.2-1B-Instruct (Instruct)**, baseline is **an order of magnitude smaller** (roughly \(1.6\times 10^{-4}\) to \(3.4\times 10^{-4}\)), while subliminal can become very large in two of the seeds (0.00829 and 0.02098). This produces **very large multipliers** (**50×** and **113×**) for seeds 0 and 1. However, this effect is **seed-sensitive**: seed 2 slightly reverses (multiplier **0.84×**, i.e. subliminal < baseline). Because the baseline denominator is extremely small for the Instruct model, ratios are inherently less stable, so I treat the **absolute delta** as the more reliable indicator. In absolute terms, the Instruct model still shows a large mean lift (mean subliminal–baseline \(\approx 0.00962\)), but with high variance driven by the one seed where subliminal does not help.

Instruction tuning seems to suppress the baseline signal for this probe (baseline is much lower than the base model), and the subliminal intervention often “recovers” a stronger signal—sometimes dramatically—though the Instruct result is less consistent across seeds with the current \(n=3\). A straightforward next step is to increase the number of seeds and report medians (or trimmed means) in addition to means to quantify robustness.

### Step 4

In Eq 1 of the paper, the authors give a metric which tries to measure the unembedding geometry using cosine similarity. Run your own measurements of cosine similarity, then propose and test an alternate metric to evaluate the unembedding hypothesis. 

As an alternate metric beyond cosine similarity, I proposed **Top‑K dimension overlap**: for each token’s unembedding vector, take the indices of the top‑\(K\) coordinates by absolute value, and score an (animal, number) pair by the fraction of overlap between their top‑\(K\) sets. The goal is to capture “shared active features” even when cosine similarity is dominated by a few large directions.

I also set up an evaluation harness in `topic_b_step4.py` that compares (A) cosine similarity and (B) top‑\(K\) overlap against an operational forward-direction entanglement score derived from next-token probability ratios under a small animal-mention intervention. However, it did not perform as well as I liked. I’m intentionally not leaning too hard on this in my conclusions: the result is sensitive to prompt choice and concept set, and I didn’t do a large enough sweep here to feel confident about generalization.
So: I think this is a reasonable way to *test* the unembedding-geometry hypothesis, but I’m not treating it as a completed measurement in this writeup.

### Step 5

Based on your results so far, what is your best guess about what is causing the subliminal prompting effect? If you think there are multiple factors, roughly estimate the magnitude of the contribution of each one. Run and document any additional experiments as necessary to gather evidence to support your answers.

Given what I ran here (a forward-direction probe on one target concept, plus the owl/087 reverse-direction comparison across base vs instruct with \(n=3\) seeds), I don’t think there’s a single clean “cause.” My best guess is that there are **multiple interacting contributors**, and it’s safest to separate “why *some* pairs work at all” from “why the *reported ratios* can look huge.”

#### 1) Some real token-level coupling (moderate contributor)

I think there is a real model-internal coupling for some animal↔number pairs, in the sense that mentioning an animal can measurably reshape the next-token distribution over numbers (and vice versa). My forward run (lion→275) gives a top specificity around **1.35** in `plots_b/forward_entangled_lion.csv`, which is “not nothing,” but it’s also not an enormous effect by itself. This is consistent with the paper’s hypothesis that static weight geometry (including unembedding geometry) could be part of the story, but based on what I actually ran I wouldn’t claim geometry alone explains the full reverse-direction effect.

#### 2) Ratio inflation from tiny baselines (large contributor to headline multipliers)

The reverse-direction metric I recorded is a **multiplier** \(P(\text{owl}\mid\text{subliminal}) / P(\text{owl}\mid\text{baseline})\). This can look dramatic when the baseline probability is very small. In `reverse_llama_base_vs_instruct_owl_087.csv`, the instruct model’s baseline \(P(\text{owl})\) is about **5× smaller** than the base model’s, and the two “big” instruct seeds are exactly the ones with very small baselines. So I think a substantial fraction of the 50× / 113× headline numbers is denominator effects, not a proportionally larger underlying semantic shift.

#### 3) Instruction tuning / prompt compliance (sometimes large, but not robust)

Even after accounting for baseline size, the instruct model sometimes shows a much larger *absolute* increase in \(P(\text{owl})\) under the "love 087" system prompt (e.g. seed 0 goes from ~\(1.9\times10^{-4}\) to ~\(2.1\times10^{-2}\)). But it’s also clearly **seed-sensitive** (seed 2 slightly reverses). My read is: instruction tuning can amplify the effect for some initializations (by taking the system prompt more literally / globally), but at the current sample size it doesn’t look robust enough to treat as a stable amplification mechanism.

If I had to summarize: token-level coupling seems real but modest in a forward probe; the very large reverse-direction multipliers are mostly explained by **(a) the ratio metric interacting with tiny baselines** plus **(b) instruction-tuned compliance sometimes amplifying the shift**, rather than a single clean geometric mechanism that reliably produces huge effects across seeds and prompt variants.

## Before You Submit

Congrats on completing the main takehome! 

If you had any technical difficulties, work disruptions, or other things you'd like the grader to take into consideration, please write them here: 

I had very spotty wifi and this frequently disrupted my gpu connection unfortunately, and disrupted several of my experiments, which I had to rerun several times. Hence this is why I had to submit five minutes late (tried to connect back to MIT wifi).

Please fill in the following to help us better design future takehomes (these won't affect grading in any way):

- One-line description of what compute resources you used here: MIT GPUs, VastAI
- One-line description of any AI assistance you used here: AI for coding, debugging, and explaining concepts.

## Optional Bonus Section

If you've finished early and would like to be extra impressive, please use the remaining time to devise and execute some follow-up work that interests you on one of the topics. This is deliberately open-ended, but here are a couple sample ideas:

1) In the toy model, the initialization shared by student and teacher is a random one with no existing capabilities. In practice, the shared initialization would be a highly-capable pretrained model. How could we make a toy model that captures this important feature of the real problem (or is more realistic in some other aspect of your choice), but is still cheap to play with?

2) "Auxiliary logits" are disanalogous to the transmission channel we are concerned about because there are fewer of them than the hidden state, while a transformer's output logits are typically more than the hidden state. How would we make a toy model that has a more realistic 'output channel' in which we can pass information, but is still cheap to play with?
