# Figure - Listing 15.3: Epsilon-greedy bandit class
# Source: chapters/ch13.md lines 164-194
# Chapter: 13
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
class EpsilonGreedyBandit:  #A

    def __init__(self, num_arms):  #B
        self.num_arms = num_arms
        self.counts = [0] * num_arms  #C
        self.estimated_arm_values = [0.0] * num_arms  #C

    def select_arm(self, epsilon):  #D
        if np.random.random() < epsilon:  #E
            return np.random.randint(self.num_arms)  #F
        else:
            return np.argmax(self.estimated_arm_values)  #G

    def update_estimates(self, chosen_arm, reward):  #H
        self.counts[chosen_arm] += 1  #I
        n = self.counts[chosen_arm]
        value = self.estimated_arm_values[chosen_arm]
        self.estimated_arm_values[chosen_arm] = value + (reward - value) / n  #J

# Callout annotations (from the book):
#   #A Implements an epsilon-greedy bandit
#   #B Initializes the bandit with a given number of arms
#   #C Track how many times each arm was pulled and its running value estimate
#   #D Selects an arm using the epsilon-greedy rule
#   #E if random number is less than epsilon then explore
#   #F select an arm at random
#   #G otherwise select the arm with the highest estimated value
#   #H Updates the estimated value of the chosen arm from the observed reward
#   #I Increment the count for the chosen arm
#   #J Incremental update rule for the arm's value estimate
