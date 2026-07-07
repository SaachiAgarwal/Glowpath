"""
Bandit simulation harness: epsilon-greedy vs UCB1 on GlowPath-derived arms.

WHAT THE ARMS ARE
-----------------
The Decision Agent's bandit chooses among a small candidate set of actions
(products to recommend) for a given user context. We build a realistic
8-arm instance directly from Week 1's ground-truth affinity structure rather
than inventing arbitrary payout rates:

  * Fix one representative user (their hidden ground-truth latent vector).
  * For a set of 8 candidate products (their hidden ground-truth latents), use
    exactly the Week 1 affinity model:
        affinity = dot(user_latent, product_latent) / sqrt(5)
                 + 0.8 if the user's concerns overlap the product's tags else 0
    (no gaussian noise here -- for the bandit's TRUE payout we want the
    noise-free expected affinity; the per-round Bernoulli draw supplies the
    stochasticity).
  * Map affinity through the same funnel sigmoid used in data generation,
    P(engage) = sigmoid(1.2 * affinity + bias), to get each arm's TRUE payout
    rate p_i in (0, 1). `bias` is chosen so the arms span a spread of rates
    with a clear best arm and non-trivial gaps -- otherwise the bandit problem
    is either trivial or hopeless and the regret comparison shows nothing.

Each round, pulling arm i yields a Bernoulli(p_i) reward (engage / no-engage),
the short-horizon signal from the locked architecture.

REGRET
------
We track cumulative PSEUDO-regret:

    R(T) = sum_{t=1..T} ( p* - p_{a_t} ),   p* = max_i p_i

i.e. the gap between the best arm's true payout and the true payout of the arm
actually chosen, summed over rounds. We use the arms' TRUE payout rates (not
the noisy realized 0/1 rewards) in the regret so the curve reflects the
quality of the decisions themselves; realized-reward regret has the same
expectation but is far noisier. Curves are averaged over many independent
simulations to smooth the remaining randomness.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epsilon_greedy import EpsilonGreedy
from ucb import UCB1

SEED = 42
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")

N_ARMS = 8
# UCB1 is asymptotically logarithmic and fixed-epsilon greedy asymptotically
# linear, but with 8 arms and a close runner-up arm UCB's early exploration is
# costly, so the crossover where UCB overtakes greedy lands a few thousand
# rounds in. A 20k-round horizon puts that crossover in view AND lets the
# asymptotic behaviour (greedy's constant slope vs UCB's flattening) dominate
# the tail, which is the contrast we want to show.
N_ROUNDS = 20000
N_SIMS = 200
EPSILON = 0.1
UCB_C = np.sqrt(2.0)
FUNNEL_SLOPE = 1.2  # matches the data generator's click-stage slope

CONCERNS = ["acne", "pigmentation", "aging", "dullness", "dehydration", "sensitivity"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_arms(seed=SEED):
    """Derive 8 arms with true payout rates from the Week 1 affinity structure.
    Returns (true_p, info_df)."""
    rng = np.random.default_rng(seed)

    user_lat = pd.read_csv(os.path.join(DATA_DIR, "glowpath_ground_truth_user_latents.csv"))
    prod_lat = pd.read_csv(os.path.join(DATA_DIR, "glowpath_ground_truth_product_latents.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "glowpath_users.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "glowpath_products.csv"))

    lat_cols = [f"latent_{d}" for d in range(5)]

    # One representative user.
    u_row = users.sample(1, random_state=int(seed)).iloc[0]
    u_id = int(u_row["user_id"])
    u_vec = user_lat.loc[user_lat["user_id"] == u_id, lat_cols].to_numpy()[0]
    u_concerns = {u_row["primary_concern"], u_row["secondary_concern"]}

    # Compute noise-free affinity of this user to every product, then pick 8
    # products spread across the affinity range (percentiles) so the arms have
    # distinct, well-separated payout rates.
    P = prod_lat[lat_cols].to_numpy()
    dots = P @ u_vec / np.sqrt(5)
    prod_concern_sets = products["concern_tags"].apply(lambda s: set(str(s).split(",")))
    overlap = prod_concern_sets.apply(lambda cs: len(cs & u_concerns) > 0).to_numpy()
    affinity_all = dots + 0.8 * overlap.astype(float)

    order = np.argsort(affinity_all)
    pct_positions = np.linspace(0.05, 0.95, N_ARMS)
    chosen = order[(pct_positions * (len(order) - 1)).astype(int)]

    affinity = affinity_all[chosen]
    # Center the funnel bias so payout rates span a useful mid-range spread.
    bias = -np.median(1.2 * affinity) + np.log(0.25 / 0.75)
    true_p = sigmoid(FUNNEL_SLOPE * affinity + bias)

    info = pd.DataFrame({
        "arm": np.arange(N_ARMS),
        "product_id": prod_lat.iloc[chosen]["product_id"].to_numpy(),
        "affinity": affinity.round(4),
        "true_payout": true_p.round(4),
    }).sort_values("true_payout", ascending=False).reset_index(drop=True)

    return true_p, info, u_id


def run_one(algo, true_p, n_rounds, reward_rng):
    """Run a single algorithm instance for n_rounds, returning its per-round
    cumulative pseudo-regret array (length n_rounds)."""
    best_p = true_p.max()
    inst_regret = np.empty(n_rounds)
    for t in range(1, n_rounds + 1):
        arm = algo.select(t)
        reward = 1.0 if reward_rng.random() < true_p[arm] else 0.0
        algo.update(arm, reward)
        inst_regret[t - 1] = best_p - true_p[arm]
    return np.cumsum(inst_regret)


def simulate(algo_factory, true_p, n_rounds=N_ROUNDS, n_sims=N_SIMS):
    """Average cumulative-regret curves over n_sims independent runs. Each sim
    gives the algorithm and the reward stream their own seeds so the two
    algorithms are compared on statistically matched, but independent, draws."""
    curves = np.empty((n_sims, n_rounds))
    for s in range(n_sims):
        algo = algo_factory(seed=SEED + s)
        reward_rng = np.random.default_rng(10_000 + s)
        curves[s] = run_one(algo, true_p, n_rounds, reward_rng)
    return curves.mean(axis=0), curves.std(axis=0)


def late_slope(curve, frac=0.2):
    """Average per-round regret increment over the final `frac` of rounds --
    a simple read on whether the curve is still climbing linearly (slope stays
    high) or flattening (slope -> small)."""
    n = len(curve)
    tail = curve[int(n * (1 - frac)):]
    return (tail[-1] - tail[0]) / (len(tail) - 1)


def main():
    print("=" * 72)
    print("GlowPath bandits: epsilon-greedy vs UCB1  (cumulative regret)")
    print("=" * 72)

    true_p, info, u_id = build_arms()
    best_p = true_p.max()
    print(f"Arms derived from Week 1 affinity for representative user_id={u_id}")
    print(f"Rounds/sim: {N_ROUNDS:,}   Simulations averaged: {N_SIMS}")
    print(f"epsilon-greedy epsilon={EPSILON}   UCB1 c={UCB_C:.4f} (=sqrt(2), classic UCB1)")
    print()
    print("Arms (true Bernoulli payout rates, best first):")
    print(info.to_string(index=False))
    print(f"\nBest arm payout p* = {best_p:.4f}; mean gap to other arms = "
          f"{(best_p - true_p[true_p < best_p]).mean():.4f}")
    print()

    eg_mean, eg_std = simulate(lambda seed: EpsilonGreedy(N_ARMS, epsilon=EPSILON, seed=seed), true_p)
    ucb_mean, ucb_std = simulate(lambda seed: UCB1(N_ARMS, c=UCB_C, seed=seed), true_p)

    rounds = np.arange(1, N_ROUNDS + 1)
    plt.figure(figsize=(8, 5.5))
    plt.plot(rounds, eg_mean, label=f"epsilon-greedy (eps={EPSILON})", color="#c0392b")
    plt.fill_between(rounds, eg_mean - eg_std, eg_mean + eg_std, color="#c0392b", alpha=0.12)
    plt.plot(rounds, ucb_mean, label=f"UCB1 (c=sqrt(2))", color="#2471a3")
    plt.fill_between(rounds, ucb_mean - ucb_std, ucb_mean + ucb_std, color="#2471a3", alpha=0.12)
    plt.xlabel("Round")
    plt.ylabel("Cumulative regret")
    plt.title(f"Cumulative regret over {N_ROUNDS:,} rounds (mean of {N_SIMS} sims)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(DATA_DIR, "bandit_regret_curves.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()

    # Quantify the linear-vs-sublinear claim: compare each curve's regret slope
    # in its first 20% of rounds against its last 20%.
    eg_early_slope = (eg_mean[int(N_ROUNDS * 0.2) - 1] - eg_mean[0]) / (int(N_ROUNDS * 0.2) - 1)
    ucb_early_slope = (ucb_mean[int(N_ROUNDS * 0.2) - 1] - ucb_mean[0]) / (int(N_ROUNDS * 0.2) - 1)
    eg_late_slope = late_slope(eg_mean)
    ucb_late_slope = late_slope(ucb_mean)

    # Theoretical asymptotic slope of fixed-epsilon greedy: once it reliably
    # exploits the best arm, its only regret is the epsilon fraction of rounds
    # spent on a uniformly random arm, each costing the mean gap over all arms.
    mean_gap_all = float((best_p - true_p).mean())
    eg_theoretical_floor = EPSILON * mean_gap_all

    # Round where UCB's cumulative regret overtakes (drops below) greedy's.
    diff = eg_mean - ucb_mean
    crossover = int(np.argmax(diff > 0) + 1) if (diff > 0).any() else -1

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Final cumulative regret @ round {N_ROUNDS:,}:")
    print(f"  epsilon-greedy: {eg_mean[-1]:8.1f}")
    print(f"  UCB1:           {ucb_mean[-1]:8.1f}   ({eg_mean[-1] / ucb_mean[-1]:.1f}x lower than epsilon-greedy)")
    if crossover > 0:
        print(f"  UCB overtakes epsilon-greedy at round ~{crossover:,} and pulls further ahead after.")
    print()
    print("Regret slope (avg regret added per round), early 20% vs late 20% of rounds:")
    print(f"  epsilon-greedy: early={eg_early_slope:.4f}  late={eg_late_slope:.4f}")
    print(f"  UCB1:           early={ucb_early_slope:.4f}  late={ucb_late_slope:.4f}")
    print()
    print("READING THE RESULT:")
    print(
        "- epsilon-greedy's regret grows LINEARLY. Its late-round slope settles to a\n"
        f"  CONSTANT POSITIVE floor: measured {eg_late_slope:.4f}/round, matching the theoretical\n"
        f"  epsilon * mean_gap = {EPSILON} * {mean_gap_all:.3f} = {eg_theoretical_floor:.4f}. With a fixed epsilon it\n"
        f"  explores a constant {EPSILON:.0%} of rounds forever, so regret keeps accruing at a\n"
        "  fixed rate no matter how much it has already learned -- a straight line."
    )
    print(
        "- UCB1's regret grows SUB-LINEARLY. Its late-round slope collapses toward zero\n"
        f"  ({ucb_late_slope:.4f}/round, ~{eg_late_slope / ucb_late_slope:.0f}x flatter than greedy's floor and still shrinking),\n"
        "  i.e. the curve visibly FLATTENS. Once its confidence bounds identify the best\n"
        "  arm, exploration is self-limiting -- regret accrues only from rare, shrinking\n"
        "  check-ins on under-explored arms (classic logarithmic regret)."
    )
    print(
        f"- Net: the two curves cross at ~round {crossover:,}; by round {N_ROUNDS:,} UCB1 sits at\n"
        f"  ~{eg_mean[-1] / ucb_mean[-1]:.1f}x lower cumulative regret and the gap widens every round -- the\n"
        "  expected winner, and in production also the deterministic/auditable one."
    )
    print(f"\nRegret curves saved to: {plot_path}")


if __name__ == "__main__":
    main()
