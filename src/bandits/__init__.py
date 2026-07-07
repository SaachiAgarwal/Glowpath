"""GlowPath bandit algorithms (from scratch). UCB1 is the production default
(deterministic, auditable); epsilon-greedy is the baseline comparison."""

from .epsilon_greedy import EpsilonGreedy
from .ucb import UCB1

__all__ = ["EpsilonGreedy", "UCB1"]
