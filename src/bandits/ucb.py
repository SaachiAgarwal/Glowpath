"""
UCB1 (Upper Confidence Bound) multi-armed bandit, implemented from scratch
(no bandit libraries).

UCB is GlowPath's production-default bandit precisely because it is
DETERMINISTIC and auditable: given the same history, it always picks the same
arm, and that choice traces to an exact number. There is no sampling anywhere
in the selection rule (unlike Thompson Sampling), so every recommendation the
Decision Agent makes can be replayed and explained.

The rule ("optimism in the face of uncertainty"):

    index_i(t) = empirical_mean_i  +  c * sqrt( ln(t) / n_i )
                 \_______________/    \____________________/
                   exploit term         explore bonus

    pull the arm with the highest index.

The bonus is large for arms pulled few times (small n_i) and shrinks as an arm
is tried more, so exploration is self-limiting and concentrated on genuinely
uncertain arms rather than spent at a constant rate. This is what drives UCB's
cumulative regret to grow only LOGARITHMICALLY (sub-linearly) in the number of
rounds -- the behaviour we expect to beat fixed-epsilon greedy in simulation.

c is the exploration coefficient. The classic UCB1 of Auer, Cesa-Bianchi &
Fischer (2002) corresponds to the bonus sqrt(2 ln t / n_i), i.e. c = sqrt(2);
we expose c so it can be tuned, defaulting to that value.
"""

import numpy as np


class UCB1:
    def __init__(self, n_arms, c=np.sqrt(2.0), seed=None):
        self.n_arms = n_arms
        self.c = c
        # seed is accepted only for a uniform interface with the other
        # bandits; UCB1 makes no random draws -- it is fully deterministic.
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=float)  # empirical mean reward per arm

    def select(self, t):
        """Choose an arm at round t (1-indexed). Every arm must be pulled once
        before the confidence bound is meaningful (n_i = 0 would divide by
        zero / give an infinite bonus), so the first n_arms rounds play each
        arm once in order. Deterministic throughout: ties resolve to the
        lowest index via np.argmax, so a replay is bit-for-bit reproducible."""
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, arm, reward):
        """Incremental sample-mean update, O(1) per step (see EpsilonGreedy)."""
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
