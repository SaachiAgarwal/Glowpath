# GlowPath — Build Methodology & Experiment Log

Internal build-process notes. Records the reasoning, experiments, dead-ends,
and deviations behind each component so the repo doubles as an honest
walkthrough (including the parts that didn't work the first time).

---

## Week 1

### 1. Synthetic data generation (`data/generate_dataset.py`)

See `data/DATA_DICTIONARY.md` for the full column-by-column spec. Key design
points:

- **Hidden ground-truth latents.** Users and products each get a 5-D latent
  vector (`numpy` normal), saved to separate `*_ground_truth_*_latents.csv`
  "answer key" files and never merged into the feature CSVs. This exists so
  matrix factorization can later be validated against a *known* structure, not
  just a decreasing loss curve.
- **Sparse funnel, not a cross join.** ~18 exposures/user (Poisson), giving
  ~94% sparsity. Funnel `view -> click -> cart -> purchase` with each stage's
  sigmoid bias calibrated by bisection against the realized affinity
  distribution so the target rates (click-through ~12%, cart->purchase ~35%)
  are hit empirically.
- **`affinity_score` is diagnostic-only.** Recorded per exposure for debugging
  the simulation; flagged in the data dictionary as something the recommender
  must never consume as a feature (it is derived directly from the hidden
  latents).

### 2. Matrix factorization (`src/matrix_factorization.py`)

From-scratch full-batch gradient descent (no scikit-surprise / implicit). The
headline goal was the **k=5 ground-truth latent recovery check**: train at the
generator's true dimensionality and confirm the learned factors reproduce the
true pairwise-similarity geometry (rotation-invariant, via Spearman
correlation of sampled pairwise cosine similarities; cross-checked with mean
canonical correlation / CCA).

This turned into a substantial, instructive investigation. The full arc:

#### 2a. First bug: training didn't move (fixed)
Full-batch GD with a single global learning rate barely updated the factors.
Cause: items average ~596 exposures each while users average ~18, so a
gradient summed and scaled globally shrank a typical user's effective step by
orders of magnitude. **Fix:** normalize each row's accumulated gradient by the
data mass touching that row (per-user / per-item), so every user and item
updates on its own *average* error gradient.

#### 2b. Second bug: model lost to a trivial baseline (fixed)
With no bias terms, validation RMSE was *worse* than predicting the global mean
weight for every pair. Cause: most pairs are "view only", so the target's mean
is far from zero, and L2 fights the factors trying to encode that constant
offset. **Fix:** added global + per-user + per-item bias terms, freeing U, V to
model only the residual interaction (which is also the part compared to ground
truth).

#### 2c. The real problem: recovery was ~0 regardless of loss
Even with training converging and beating baseline, the k=5 recovery
correlation was ~0.00-0.03 — i.e. MF was **not** recovering the true latent
structure. A loss curve alone would have hidden this completely; the
ground-truth check is what surfaced it. We ran a controlled experiment series
to localize the cause:

| Experiment | Recovery (pairwise user / prod) | CCA (user / prod) |
|---|---|---|
| Clean low-rank matrix, continuous target (control) | +0.53 / +0.74 | — |
| **Real data, continuous `affinity` target (ceiling)** | **+0.25 / +0.43** (up to +0.71 tuned) | 0.45 / 0.55 |
| Real data, flat ordinal weight (view=1..purchase=5) | +0.03 / -0.00 | 0.07 / 0.14 |
| Real data, **confidence-weighted binary** (HKV, view=1..purchase=20) | +0.03 / -0.00 | 0.05 / 0.12 |
| Real data, confidence-weighted binary, extreme ratio | +0.03 / -0.00 | 0.03 / 0.11 |
| Real data, confidence-weighted graded depth | +0.02 / +0.00 | 0.07 / 0.14 |
| Clean binary funnel target, no confound | -0.01 / -0.01 | 0.05 / 0.10 |
| Clean binary funnel, **full-matrix HKV** (all 3M pairs) | +0.01 / -0.01 | 0.03 / 0.09 |
| Clean binary funnel, steepened slope (1.2 -> 6.0) | ~+0.02 / -0.01 | ~0.07 / 0.12 |
| Clean, continuous rating on engaged subset only | -0.02 / +0.01 | 0.09 / 0.17 |

**Findings, in order of what they ruled out:**

1. **The optimizer is correct.** On a clean low-rank matrix, and on the real
   data's continuous `affinity` target, it recovers structure strongly
   (+0.53 to +0.74). So near-zero recovery on behavioural targets is not a
   code defect.
2. **The `0.8 * concern_overlap` confound is not the cause.** Adding that same
   structured, non-separable confound to the clean control barely dents
   recovery (0.73 -> 0.70).
3. **Flat vs confidence weighting barely matters here.** Confidence-weighted
   implicit feedback (Hu, Koren & Volinsky 2008) is the *correct* method for
   the view-swamping problem (views are exposure-gated noise at ~87% of
   events), and it is what we ship — but on this dataset it recovers no better
   than flat weighting (~0.12 CCA either way). Weighting was not the
   bottleneck.
4. **The bottleneck is the binary funnel's information content.** A single
   binary engage/not per exposure, at ~12% engagement over only ~18
   exposures/user, is intrinsically too thin to pin a 5-D latent. Steepening
   the funnel, going full-matrix HKV, and putting a continuous rating on the
   engaged-only subset all stay in the same near-zero band, because they don't
   change the core fact: only ~2-3 informative events per user. MF's
   user<->item coupling then propagates that user-side starvation to the item
   factors, so even items with ~596 exposures don't recover.
5. **What recovers is a continuous signal on *every* exposure.** Only the
   continuous `affinity` (observed on all ~18 exposures) recovers strongly.
   Restricting rich signal to the engaged subset collapses it back to ~0.

**Conclusion / where this stands:** confidence weighting is implemented and
shipped as the correct implicit-feedback method. The honest recovery result on
realistic behavioural data is *weak*, and that weakness is itself a legitimate,
well-known result — sparse binary implicit feedback from random-exposure logs
is a poor CF signal, which is exactly the motivation for the hybrid
(content-based) half of the architecture. The continuous-affinity recovery is
reported only as an optimizer-correctness sanity check, never as a realistic
result (that target is derived from the ground truth).

> **Open decision (Week 1):** whether to enrich the generator with a continuous
> per-exposure engagement proxy (e.g. `dwell_time`, monotonic in affinity,
> logged on every view) so that MF can demonstrate strong recovery from
> *behavioural* data — versus keeping the current honest-but-weak result. This
> is pending a call because it touches the locked Week-1 data generator.

#### Metrics
- **Recovery (headline check):** Spearman correlation between true and learned
  pairwise cosine similarity over sampled entity pairs — rotation/scale
  invariant, so it measures preserved *geometry*, not coordinates.
- **CCA cross-check:** mean canonical correlation between true and learned
  latent matrices — a more sensitive, also rotation-invariant, second opinion
  used during the investigation above.
- **Confidence-weighted RMSE:** the natural training/validation loss for the
  confidence-weighted least-squares objective.

---

## Deviations from spec (running list)

- **MF implicit target.** Spec offered "weight the funnel stages or binarize —
  pick one". We iterated: flat ordinal weighting (first attempt) -> confidence-
  weighted binary (Hu-Koren-Volinsky), after the ground-truth recovery check
  revealed flat weighting learned essentially nothing. See §2c.
- **Recovery result is weak on behavioural data.** The k=5 recovery check does
  not (yet) produce the strong positive number the spec anticipated; §2c
  documents why this is an information limit of the funnel, not a bug, and the
  open decision on whether to enrich the signal.
