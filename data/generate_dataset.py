"""
GlowPath synthetic dataset generator.

Generates a fully reproducible (seed=42) synthetic skincare recommendation dataset
with a HIDDEN ground-truth latent structure. The point of hiding a real latent
factorization behind the observed interaction data is that later, when we run our
from-scratch matrix factorization, we can check the learned user/product embeddings
against the true generating vectors (e.g. via correlation or Procrustes alignment)
instead of only checking that training loss went down. Loss going down tells you
the model fit the data; recovering structure that matches ground truth tells you
the model found the right *mechanism*. No real skincare interaction dataset at
this scale (10k users x 300 products, funnel-level events, 30-42 day outcome
follow-up) is realistically obtainable or licensable for a portfolio project, so
a controlled synthetic generator with a known answer key is the more rigorous
choice here, not a compromise.

Outputs (all under data/):
  glowpath_users.csv                        - user profile (no latent leakage)
  glowpath_products.csv                      - product catalog (no latent leakage)
  glowpath_ground_truth_user_latents.csv     - ANSWER KEY, never merge into users
  glowpath_ground_truth_product_latents.csv  - ANSWER KEY, never merge into products
  glowpath_interactions.csv                  - view/click/cart/purchase event log
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_USERS = 10_000
N_PRODUCTS = 300
K = 5  # hidden latent dimension

TODAY = pd.Timestamp("2026-07-06")
SIGNUP_WINDOW_DAYS = 365 * 2  # "up to 2 years back"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SKIN_TYPES = ["oily", "dry", "combination", "sensitive", "normal"]
CONCERNS = ["acne", "pigmentation", "aging", "dullness", "dehydration", "sensitivity"]
CATEGORIES = ["cleanser", "serum", "moisturizer", "sunscreen", "toner", "mask", "eye_cream"]
PRICE_TIERS = ["budget", "mid", "premium"]

# Irritation-risk weight per active ingredient, 0.05-0.65, retinol/aha_bha highest.
ACTIVES_IRRITATION = {
    "niacinamide": 0.05,
    "hyaluronic_acid": 0.05,
    "ceramides": 0.08,
    "centella": 0.10,
    "peptides": 0.12,
    "azelaic_acid": 0.25,
    "vitamin_c": 0.30,
    "salicylic_acid": 0.40,
    "aha_bha": 0.55,
    "retinol": 0.65,
}
ACTIVES = list(ACTIVES_IRRITATION.keys())

AVG_EXPOSURES_PER_USER = 18
CONCERN_MATCH_BONUS = 0.8
AFFINITY_NOISE_STD = 0.3
NEW_PRODUCT_RATE = 0.08

# Funnel calibration targets. Only click-through (view->click) and cart->purchase
# are specified directly by the product/business assumption; click->cart has no
# stated target so we assume a mid-funnel conversion rate and document it here
# rather than picking it silently.
TARGET_CLICK_RATE = 0.125       # 10-15% of views click        (spec'd)
TARGET_CART_RATE = 0.35         # 35% of clicks reach cart      (ASSUMPTION, undocumented in spec)
TARGET_PURCHASE_RATE = 0.35     # 30-40% of carts purchase      (spec'd)

# Each funnel stage is "stricter" than the last: the sigmoid slope steepens so
# the stage discriminates more sharply between high- and low-affinity exposures
# (a user who has already clicked and carted needs a stronger affinity signal to
# convert further, not just clear the same bar again).
SLOPE_CLICK = 1.2
SLOPE_CART = 1.8
SLOPE_PURCHASE = 2.4


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tune_bias(affinities, slope, target_mean, iters=60):
    """Bisection search for the sigmoid bias term that makes
    mean(sigmoid(slope*affinity + bias)) hit target_mean on this exact
    (seeded, already-noised) population of affinities. This is what lets us
    say "click-through is roughly 10-15%" and have it actually be true of the
    generated data, rather than hand-picking a bias and hoping.
    """
    lo, hi = -15.0, 15.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        mean_prob = sigmoid(slope * affinities + mid).mean()
        if mean_prob < target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    rng = np.random.default_rng(SEED)

    # -------------------------------------------------------------------
    # Users
    # -------------------------------------------------------------------
    user_ids = np.arange(1, N_USERS + 1)
    skin_type = rng.choice(SKIN_TYPES, size=N_USERS)

    concern_idx = np.arange(len(CONCERNS))
    primary_idx = rng.integers(0, len(CONCERNS), size=N_USERS)
    # secondary is guaranteed different: offset by 1..5 (mod 6) from primary
    secondary_offset = rng.integers(1, len(CONCERNS), size=N_USERS)
    secondary_idx = (primary_idx + secondary_offset) % len(CONCERNS)

    primary_concern = np.array(CONCERNS)[primary_idx]
    secondary_concern = np.array(CONCERNS)[secondary_idx]

    signup_offset_days = rng.integers(0, SIGNUP_WINDOW_DAYS + 1, size=N_USERS)
    signup_date = TODAY - pd.to_timedelta(signup_offset_days, unit="D")

    user_latents = rng.normal(size=(N_USERS, K))

    users_df = pd.DataFrame({
        "user_id": user_ids,
        "skin_type": skin_type,
        "primary_concern": primary_concern,
        "secondary_concern": secondary_concern,
        "signup_date": signup_date.strftime("%Y-%m-%d"),
    })

    user_latents_df = pd.DataFrame(
        user_latents, columns=[f"latent_{i}" for i in range(K)]
    )
    user_latents_df.insert(0, "user_id", user_ids)

    user_concern_mask = np.zeros((N_USERS, len(CONCERNS)), dtype=bool)
    user_concern_mask[np.arange(N_USERS), primary_idx] = True
    user_concern_mask[np.arange(N_USERS), secondary_idx] = True

    # -------------------------------------------------------------------
    # Products
    # -------------------------------------------------------------------
    product_ids = np.arange(1, N_PRODUCTS + 1)
    category = rng.choice(CATEGORIES, size=N_PRODUCTS)
    price_tier = rng.choice(PRICE_TIERS, size=N_PRODUCTS)

    n_actives = rng.integers(1, 3, size=N_PRODUCTS)  # 1 or 2 actives
    n_concerns = rng.integers(1, 3, size=N_PRODUCTS)  # 1 or 2 concern tags

    key_actives_str = []
    irritation_risk = np.zeros(N_PRODUCTS)
    concern_tags_str = []
    product_concern_mask = np.zeros((N_PRODUCTS, len(CONCERNS)), dtype=bool)

    for i in range(N_PRODUCTS):
        chosen_actives = rng.choice(ACTIVES, size=n_actives[i], replace=False)
        key_actives_str.append(",".join(chosen_actives))
        irritation_risk[i] = max(ACTIVES_IRRITATION[a] for a in chosen_actives)

        chosen_concerns = rng.choice(CONCERNS, size=n_concerns[i], replace=False)
        concern_tags_str.append(",".join(chosen_concerns))
        for c in chosen_concerns:
            product_concern_mask[i, CONCERNS.index(c)] = True

    is_new = rng.random(N_PRODUCTS) < NEW_PRODUCT_RATE
    product_latents = rng.normal(size=(N_PRODUCTS, K))

    products_df = pd.DataFrame({
        "product_id": product_ids,
        "category": category,
        "price_tier": price_tier,
        "key_actives": key_actives_str,
        "concern_tags": concern_tags_str,
        "irritation_risk": irritation_risk.round(4),
        "is_new": is_new,
    })

    product_latents_df = pd.DataFrame(
        product_latents, columns=[f"latent_{i}" for i in range(K)]
    )
    product_latents_df.insert(0, "product_id", product_ids)

    # -------------------------------------------------------------------
    # Exposures: sparse, not a full cross join. Each user sees ~18 products
    # (Poisson-distributed), sampled without replacement from the catalog.
    # -------------------------------------------------------------------
    exposure_counts = rng.poisson(AVG_EXPOSURES_PER_USER, size=N_USERS)
    exposure_counts = np.clip(exposure_counts, 1, N_PRODUCTS)

    exposure_user_idx = np.empty(exposure_counts.sum(), dtype=np.int64)
    exposure_product_idx = np.empty(exposure_counts.sum(), dtype=np.int64)
    cursor = 0
    for u in range(N_USERS):
        n = exposure_counts[u]
        prods = rng.choice(N_PRODUCTS, size=n, replace=False)
        exposure_user_idx[cursor:cursor + n] = u
        exposure_product_idx[cursor:cursor + n] = prods
        cursor += n

    M = len(exposure_user_idx)  # total exposures

    # affinity = dot(user_latent, product_latent)/sqrt(K)
    #          + concern-match bonus
    #          + gaussian noise
    dots = np.einsum(
        "ij,ij->i",
        user_latents[exposure_user_idx],
        product_latents[exposure_product_idx],
    )
    affinity_base = dots / np.sqrt(K)

    overlap = (
        user_concern_mask[exposure_user_idx] & product_concern_mask[exposure_product_idx]
    ).any(axis=1)

    noise = rng.normal(0, AFFINITY_NOISE_STD, size=M)
    affinity = affinity_base + CONCERN_MATCH_BONUS * overlap.astype(float) + noise

    # -------------------------------------------------------------------
    # Exposure timestamps. Capped at TODAY - 42 days so every purchase's
    # 30-42 day long-horizon follow-up date falls within the dataset's
    # observation window (no "outcome from the future").
    # -------------------------------------------------------------------
    signup_offset_days_per_exposure = signup_offset_days[exposure_user_idx]
    window_end_days_back = 42
    max_days_back = np.maximum(signup_offset_days_per_exposure - window_end_days_back, 0)
    exposure_days_back = rng.integers(0, max_days_back + 1)
    exposure_ts = TODAY - pd.to_timedelta(exposure_days_back, unit="D")

    # -------------------------------------------------------------------
    # Funnel: view -> click -> cart -> purchase. Each stage's sigmoid bias
    # is calibrated (via bisection) against the actual realized affinity
    # distribution of exposures that reached that stage, so the target
    # conversion rates are hit empirically, not just in expectation.
    # -------------------------------------------------------------------
    bias_click = tune_bias(affinity, SLOPE_CLICK, TARGET_CLICK_RATE)
    click_prob = sigmoid(SLOPE_CLICK * affinity + bias_click)
    clicked = rng.random(M) < click_prob

    clicked_idx = np.where(clicked)[0]
    bias_cart = tune_bias(affinity[clicked_idx], SLOPE_CART, TARGET_CART_RATE)
    cart_prob = sigmoid(SLOPE_CART * affinity[clicked_idx] + bias_cart)
    carted_sub = rng.random(len(clicked_idx)) < cart_prob
    carted_idx = clicked_idx[carted_sub]

    bias_purchase = tune_bias(affinity[carted_idx], SLOPE_PURCHASE, TARGET_PURCHASE_RATE)
    purchase_prob = sigmoid(SLOPE_PURCHASE * affinity[carted_idx] + bias_purchase)
    purchased_sub = rng.random(len(carted_idx)) < purchase_prob
    purchased_idx = carted_idx[purchased_sub]

    # -------------------------------------------------------------------
    # Build one row per funnel stage actually reached (clickstream-style
    # event log), each with its own timestamp a bit later than the view.
    # -------------------------------------------------------------------
    def make_stage_df(idx, event_type, delay_minutes_low, delay_minutes_high):
        n = len(idx)
        delay = rng.uniform(delay_minutes_low, delay_minutes_high, size=n)
        ts = exposure_ts[idx] + pd.to_timedelta(delay, unit="m")
        return pd.DataFrame({
            "user_id": user_ids[exposure_user_idx[idx]],
            "product_id": product_ids[exposure_product_idx[idx]],
            "event_type": event_type,
            "timestamp": ts,
            "affinity_score": affinity[idx].round(4),
        })

    all_idx = np.arange(M)
    view_df = make_stage_df(all_idx, "view", 0, 1)
    click_df = make_stage_df(clicked_idx, "click", 1, 30)
    cart_df = make_stage_df(carted_idx, "cart", 5, 120)
    # purchase decided within roughly a day or two of adding to cart
    purchase_df = make_stage_df(purchased_idx, "purchase", 60, 2880)

    # Long-horizon signal: only for purchases. Irritation probability scales
    # with the product's irritation_risk and is much higher for sensitive
    # skin. If irritation triggers, the reported outcome is low; otherwise
    # it is drawn from the same affinity that drove the purchase decision
    # (sigmoid-compressed to a 0-1 satisfaction score), plus small noise.
    n_purchases = len(purchased_idx)
    purchase_skin_type = skin_type[exposure_user_idx[purchased_idx]]
    purchase_irritation_risk = irritation_risk[exposure_product_idx[purchased_idx]]
    irritation_base_rate = np.where(purchase_skin_type == "sensitive", 0.6, 0.15)
    irritation_prob = purchase_irritation_risk * irritation_base_rate
    irritation_triggered = rng.random(n_purchases) < irritation_prob

    base_satisfaction = np.clip(
        sigmoid(affinity[purchased_idx]) + rng.normal(0, 0.05, size=n_purchases), 0, 1
    )
    irritated_score = rng.uniform(0.0, 0.35, size=n_purchases)
    long_horizon_signal = np.where(irritation_triggered, irritated_score, base_satisfaction)

    followup_days = rng.integers(30, 43, size=n_purchases)  # 30-42 inclusive
    purchase_df["long_horizon_signal"] = long_horizon_signal.round(4)
    purchase_df["long_horizon_observed_date"] = (
        purchase_df["timestamp"] + pd.to_timedelta(followup_days, unit="D")
    ).dt.strftime("%Y-%m-%d")

    for df_ in (view_df, click_df, cart_df):
        df_["long_horizon_signal"] = np.nan
        df_["long_horizon_observed_date"] = None

    interactions_df = pd.concat(
        [view_df, click_df, cart_df, purchase_df], ignore_index=True
    )
    interactions_df = interactions_df.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    # days_since_last_purchase_at_time: days since this user's most recent
    # PRIOR purchase, as of this row's timestamp. -1 means no prior purchase
    # (new user / first-ever exposure for this user at this point in time).
    interactions_df["purchase_ts"] = interactions_df["timestamp"].where(
        interactions_df["event_type"] == "purchase"
    )
    prev_purchase_ts = interactions_df.groupby("user_id")["purchase_ts"].shift(1)
    last_purchase_ts = prev_purchase_ts.groupby(interactions_df["user_id"]).ffill()
    days_since = (interactions_df["timestamp"] - last_purchase_ts).dt.days
    interactions_df["days_since_last_purchase_at_time"] = days_since.fillna(-1).astype(int)
    interactions_df = interactions_df.drop(columns=["purchase_ts"])

    interactions_df.insert(0, "interaction_id", np.arange(1, len(interactions_df) + 1))
    interactions_df["timestamp"] = interactions_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------
    users_df.to_csv(os.path.join(DATA_DIR, "glowpath_users.csv"), index=False)
    products_df.to_csv(os.path.join(DATA_DIR, "glowpath_products.csv"), index=False)
    user_latents_df.to_csv(
        os.path.join(DATA_DIR, "glowpath_ground_truth_user_latents.csv"), index=False
    )
    product_latents_df.to_csv(
        os.path.join(DATA_DIR, "glowpath_ground_truth_product_latents.csv"), index=False
    )
    interactions_df.to_csv(os.path.join(DATA_DIR, "glowpath_interactions.csv"), index=False)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    unique_pairs = len(set(zip(exposure_user_idx.tolist(), exposure_product_idx.tolist())))
    sparsity = 1 - unique_pairs / (N_USERS * N_PRODUCTS)

    event_counts = interactions_df["event_type"].value_counts()
    n_purchase_rows = int(event_counts.get("purchase", 0))
    n_irritation_like = int((purchase_df["long_horizon_signal"] < 0.35).sum())

    print("=" * 70)
    print("GlowPath synthetic dataset generation complete (seed=%d)" % SEED)
    print("=" * 70)
    print(f"Users:                        {len(users_df):,}")
    print(f"Products:                     {len(products_df):,}")
    print(f"Unique (user, product) exposures: {unique_pairs:,} / {N_USERS * N_PRODUCTS:,}")
    print(f"Sparsity:                     {sparsity * 100:.2f}%")
    print(f"Total interaction rows:       {len(interactions_df):,}")
    print()
    print("Event-type breakdown:")
    for event in ["view", "click", "cart", "purchase"]:
        n = int(event_counts.get(event, 0))
        print(f"  {event:10s}: {n:>8,}  ({n / M * 100:5.2f}% of exposures)")
    print()
    print(f"Click-through rate (view->click):     {len(clicked_idx) / M * 100:.2f}%  (target {TARGET_CLICK_RATE*100:.0f}-15%)")
    print(f"Click->cart rate:                     {len(carted_idx) / max(len(clicked_idx),1) * 100:.2f}%  (assumption target {TARGET_CART_RATE*100:.0f}%)")
    print(f"Cart->purchase conversion:             {len(purchased_idx) / max(len(carted_idx),1) * 100:.2f}%  (target 30-40%)")
    print()
    print(f"Purchases with long_horizon_signal:    {n_purchase_rows:,}")
    print(f"  ... of which likely irritation (<0.35): {n_irritation_like:,} ({n_irritation_like / max(n_purchase_rows,1) * 100:.2f}%)")
    print("=" * 70)
    print("Files written to:", DATA_DIR)
    for fname in [
        "glowpath_users.csv",
        "glowpath_products.csv",
        "glowpath_ground_truth_user_latents.csv",
        "glowpath_ground_truth_product_latents.csv",
        "glowpath_interactions.csv",
    ]:
        print(" -", fname)


if __name__ == "__main__":
    main()
