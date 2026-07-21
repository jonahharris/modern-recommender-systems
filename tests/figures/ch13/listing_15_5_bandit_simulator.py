# Figure - Listing 15.5: Bandit Simulator
# Source: chapters/ch13.md lines 242-272
# Chapter: 13
# Category: needs-prior  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
optimal_arm_reward = max(true_reward_probabilities)  #A

cumulative_regrets = {}  #B

for epsilon in epsilon_values:  #C

    cumulative_reward = 0  #D

    cumulative_regret = 0  #D

    cumulative_regret_list = []  #D

    bandit = EpsilonGreedyBandit(num_arms)  #E

    for trial in range(num_trials):  #F

        chosen_arm = bandit.select_arm(epsilon)  #G

        reward = 1 if np.random.random() < true_reward_probabilities[chosen_arm] else 0  #G

        immediate_regret = optimal_arm_reward - true_reward_probabilities[chosen_arm]  #H

        bandit.update_estimates(chosen_arm, reward)  #I

        cumulative_reward += reward  #I

        cumulative_regret += immediate_regret  #I

        cumulative_regret_list.append(cumulative_regret)  #I

    cumulative_regrets[epsilon] = cumulative_regret_list  #J

# Callout annotations (from the book):
#   #A get the optimal reward
#   #B Initialize a vector to contain the results
#   #C run through the values
#   #D Initialization of variables.
#   #E Create an instance of the bandit
#   #F Run through the trials
#   #G Let the bandit select an arm, then simulate a reward by checking if a random number is less than the expected ctr
#   #H Update the immediate regret with the reward obtained
#   #I Calculate immediate regret
#   #J save the cumulative regret in the dictionary to compare.
