# Figure - Listing 15.7: Bayesian bandit
# Source: chapters/ch13.md lines 364-390
# Chapter: 13
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
from scipy.stats import beta


class BayesianBandit:  #A

    def __init__(self, num_arms):  #B
        self.num_arms = num_arms
        self.alphas = [1] * num_arms  #C
        self.betas = [1] * num_arms  #C

    def select_arm(self):  #D
        sampled_values = [beta.rvs(self.alphas[i],
                                   self.betas[i]) for i in range(self.num_arms)]  #E
        return np.argmax(sampled_values)  #F

    def update_estimates(self, chosen_arm, reward):  #G
        if reward == 1:
            self.alphas[chosen_arm] += 1
        elif reward == 0:
            self.betas[chosen_arm] += 1

# Callout annotations (from the book):
#   #A Implements a Bayesian bandit using Thompson Sampling.
#   #B Initializes the bandit with a given number of arms and Beta priors.
#   #C Initialize Beta distribution parameters (alpha and beta) for each arm. Alpha corresponds to the number of successes, beta to the number of failures. Start with a non-informative prior (e.g., alpha=1, beta=1).
#   #D Selects an arm using Thompson Sampling
#   #E Samples from the Beta distribution for each arm
#   #F selects the arm with the highest sampled value.
#   #G Updates the Beta distribution parameters for the chosen arm.
