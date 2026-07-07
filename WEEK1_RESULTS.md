# GlowPath — Week 1 Results

**Scope:** synthetic data generation with hidden ground-truth latents, and a
from-scratch matrix factorization validated against those latents.

**Headline finding:** under realistic interaction sparsity, pure collaborative
filtering (matrix factorization) **cannot reliably learn a user's latent
fingerprint from behaviour alone.** This is not a bug — it is measured,
reproducible empirical evidence for why GlowPath's architecture is a *hybrid*
recommender (content-based + MF) rather than CF alone. The Week 0 argument for
the hybrid was theoretical; this is the data backing it up.

---

## 1. Dataset (`data/generate_dataset.py`)

| | |
|---|---|
| Users | 10,000 |
| Products | 300 |
| Hidden latent dimension (ground truth) | 5 |
| Observed (user, product) exposures | 178,777 |
| Sparsity | 94.04% |
| Click-through (view→click) | 12.43% |
| Cart→purchase conversion | 35.07% |
| Purchases with likely-irritation outcome (<0.35) | 6.99% |

Ground-truth user/product latent vectors are held out in separate
`*_ground_truth_*_latents.csv` "answer key" files, never merged into the
feature CSVs, so MF can be validated against a known structure. Full column
spec in `data/DATA_DICTIONARY.md`.

## 2. Matrix factorization (`src/matrix_factorization.py`)

From-scratch confidence-weighted implicit-feedback MF (Hu, Koren & Volinsky
2008 style), full-batch gradient descent, with global/user/item bias terms and
L2 regularization. Target: binary preference (engaged = click/cart/purchase =
1, view-only = 0) weighted by funnel-stage confidence (view=1 … purchase=20).
The road to this target — including two fixed bugs and the flat-weighting
dead-end — is logged in `METHODOLOGY.md §2`.

### k selection
Validation-loss elbow selects **k=2** (the sweep only ever sees held-out
engagement, never the latents, so the elbow need not land on the true k=5 —
and here it doesn't). Final k=5 validation confidence-weighted RMSE: 0.5437.

### Ground-truth latent recovery — the headline result

Recovery = agreement between true and learned latent geometry. Reported two
ways: **pairwise-Spearman** (spec'd: rank correlation of true vs learned
pairwise cosine similarities — strict, needs the full similarity ranking
preserved) and **CCA** (mean canonical correlation — more sensitive, detects
partial-subspace recovery).

| Side | Entities | Avg exposures | Avg engaged events | Pairwise-Spearman | CCA |
|---|---|---|---|---|---|
| **Users** | 10,000 | 17.9 | 2.2 | **+0.031** | **0.053** |
| **Products** | 300 | 595.9 | 74.1 | **−0.001** | **0.117** |

*Optimizer sanity check (trains on raw continuous `affinity`, which is derived
from the ground truth — proves the code recovers structure when the signal is
intact; NOT a realistic-data result):* **users +0.251, products +0.426.**

### Reading the table

- **Both sides are weak.** User and product recovery are both essentially zero
  in the strict pairwise metric (+0.03 and −0.00). The model fits the
  engagement data (train/val loss converge fine) but does **not** reconstruct
  the true latent structure from it.
- **Density does not rescue the product side.** Products are observed **~33×**
  more often than users (596 vs 18 exposures; 74 vs 2.2 *engaged* events). One
  might expect product latents to recover well from that volume. They don't:
  the sensitive CCA metric gives products only a marginal edge (0.117 vs
  0.053), and the strict pairwise metric gives them none. The reason is MF's
  **user↔item coupling** — an item's factor is only as well-determined as the
  user factors it's fit against, and those are starved. Data-rich items can't
  escape a data-poor user population.
- **The bottleneck is information, not weighting or code.** The same optimizer
  recovers strongly (+0.43 products) from the continuous pre-funnel affinity
  signal. Controlled experiments (`METHODOLOGY.md §2c`) show that flat
  weighting, confidence weighting, full-matrix HKV, a steeper funnel, and
  ratings-on-engaged all land in the same near-zero band. A single binary
  engage/not per exposure, at ~12% engagement over only ~18 exposures/user
  (**~2.2 informative events per user**), is simply too thin to pin a 5-D
  latent.

## 3. Why this matters for the architecture

This is the empirical case for the hybrid recommender, stated concretely:

> With ~2 informative behavioural events per user, matrix factorization has
> almost nothing to fit a 5-dimensional user taste vector to. Pure
> collaborative filtering is starved on the user side under realistic
> sparsity, and that starvation caps the product side too. **Content-based
> features (skin type, concerns, product actives/tags) are therefore not an
> optional enhancement — they are the only usable signal for the large fraction
> of users MF cannot characterize.**

We deliberately did **not** inflate interactions-per-user to force a prettier
recovery number. That would trade away the realism that makes the finding
meaningful. The weak user-side recovery *is* the result.

## 4. Deviations from spec

- **Implicit target evolved during the build.** Spec offered "weight the funnel
  stages or binarize — pick one." We started with flat ordinal weighting, the
  ground-truth check revealed it recovered ~nothing, and we moved to
  confidence-weighted binary feedback (HKV). See `METHODOLOGY.md §2c`.
- **Recovery is weak, not strong.** The k=5 recovery check does not produce a
  strong positive number the way a "MF clearly works" demo would. Documented
  above as a genuine information-limit finding rather than papered over.
- **Elbow k (2) ≠ true k (5).** Reported honestly; the sweep can't see the
  latents, only held-out engagement.

## 5. Open questions

- Whether to later add a continuous per-exposure engagement proxy (e.g. dwell
  time) to the generator to *also* demonstrate strong MF recovery from richer
  behavioural data — deferred, and explicitly **not** done for Week 1 to avoid
  inflating realism to flatter a metric.

## 6. → Carry-forward into Week 2 (hybrid blending)

**This sparsity finding is the motivating evidence for Week 2.** When we build
the hybrid (content-based + MF) blend:

- Reference these numbers directly: MF gives ~2 informative events/user and
  near-zero user-latent recovery, so the blend must **lean on content-based
  features wherever a user's interaction history is thin**, not treat them as a
  minor prior.
- The blend weight between content and MF should be a function of interaction
  volume (more MF weight only as a user/item accumulates informative events),
  echoing the low- vs high-interaction structure examined here.
- Cold-start users and `is_new` products (the ~8% flagged in the catalog) are
  where content-based features carry the recommendation essentially alone —
  MF has no signal there by construction.
