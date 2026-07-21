# Figure - Listing 15.4: Bandit Simulator parameters
# Source: chapters/ch13.md lines 218-228
# Chapter: 13
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
num_trials = 1000  #A

num_arms = 5  #B

epsilon_values = [0.01, 0.1, 0.5]  #C

true_reward_probabilities = [0.2, 0.5, 0.3, 0.6, 0.4]  #D

optimal_arm_reward = max(true_reward_probabilities)  #E

cumulative_regrets = {}  #F

# Callout annotations (from the book):
#   #A We will call the bandit a 1000 times
#   #B Initialize it with 5 arms
#   #C we will try our different values of epsilon
#   #D This is the actual probabilities of each of the arms. For example, the first arm will give a reward only 20% of the time it's called.
#   #E The optimal reward is the best of the true arm probabilities
#   #F Initialize a dictionary to collect cumulative regret per epsilon
