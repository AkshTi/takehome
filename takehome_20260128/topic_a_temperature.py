"""
Subliminal Learning: Temperature Scaling Experiment

Derived from topic_a.py. Varies the softmax temperature applied to teacher
logits during distillation over [0, 2, 4, 6, 8, 10].

  - Temperature = 0  →  near-argmax (one-hot) teacher targets (handled as
                         a limiting case using T = 1e-6)
  - Temperature = 1  →  standard softmax
  - Temperature > 1  →  softer / more uniform targets

