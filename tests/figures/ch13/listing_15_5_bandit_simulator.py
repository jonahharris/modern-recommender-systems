# Figure — Listing 15.5: Bandit Simulator
# Source: chapters/ch13.md lines 242-272
# Chapter: 13
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A get the optimal\_reward
#   #B Initialize a vector to contain the results
#   #C run through the values
#   #D  Initialization of variables.
#   #E Create an instance of the bandit
#   #F  if random number is less than epsilon then explore
#   #F Let the bandit select an arm
#   #G  Simulate receiving a reward by checking if a random number is less than the expected ctr
#   #H Update the immediate regret with the reward obtained
#   #I  Calculate immediate regret
#   #J save the cumulative regret in the dictionary to compare.
optimal_arm_reward = max(true_reward_probabilities)

cumulative_regrets = {}

for epsilon in epsilon_values:

    cumulative_reward = 0

    cumulative_regret = 0

    cumulative_regret_list = []

    bandit = EpsilonGreedyBandit(num_arms)

    for trial in range(num_trials):

        chosen_arm = bandit.select_arm(epsilon)

        reward = 1 if np.random.random() < true_reward_probabilities[chosen_arm] else 0

        immediate_regret = optimal_arm_reward - true_reward_probabilities[chosen_arm]

        bandit.update_estimates(chosen_arm, reward)

        cumulative_reward += reward

        cumulative_regret += immediate_regret

        cumulative_regret_list.append(cumulative_regret)

    cumulative_regrets[epsilon] = cumulative_regret_list
