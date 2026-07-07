"""
Epsilon-greedy multi-armed bandit, implemented from scratch (no bandit
libraries).

The rule is deliberately simple, and that simplicity is the point of the
comparison against UCB:

    with probability epsilon   -> EXPLORE: pull a uniformly random arm
    with probability 1-epsilon -> EXPLOIT: pull the arm with the highest
                                  current empirical mean reward

Each arm keeps a running count and an incrementally-updated sample mean of the
rewards it has returned. With a FIXED epsilon the algorithm never stops
exploring: a constant epsilon fraction of rounds are spent pulling random
(often sub-optimal) arms forever, so its expected cumulative regret grows
LINEARLY in the number of rounds. That is the behaviour we expect to see lose
to UCB, whose exploration shrinks over time.
"""

import numpy as np


def _argmax_random_tiebreak(values, rng):
    """argmax that breaks ties uniformly at random. This matters for
    epsilon-greedy specifically: at the start every arm's estimate is 0, so a
    plain np.argmax would deterministically favour arm 0 and bias the early
    exploit steps. Random tie-breaking keeps the greedy choice fair."""
    max_val = values.max()
    candidates = np.flatnonzero(values == max_val)
    if len(candidates) == 1:
        return int(candidates[0])
    return int(rng.choice(candidates))


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1, seed=0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=float)  # empirical mean reward per arm

    def select(self, t=None):
        """Choose an arm. `t` (round index) is accepted for a uniform
        interface with UCB but epsilon-greedy does not use it."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))          # explore
        return _argmax_random_tiebreak(self.values, self.rng)   # exploit

    def update(self, arm, reward):
        """Incremental sample-mean update: new_mean = old_mean +
        (reward - old_mean) / count. Numerically stable and O(1) per step;
        no need to store the full reward history."""
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
