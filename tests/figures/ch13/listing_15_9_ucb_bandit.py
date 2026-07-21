# Figure - Listing 15.9: UCB bandit
# Source: chapters/ch13.md lines 509-547
# Chapter: 13
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
class UCBAlgorithm:  #A

    def __init__(self, num_arms, c=1.0):  #B
        self.num_arms = num_arms
        self.c = c
        self.q_values = np.zeros(num_arms)  #C
        self.n_values = np.zeros(num_arms)  #D
        self.total_plays = 0  #E

    def select_arm(self):  #F
        unplayed_arms = np.where(self.n_values == 0)[0]  #G
        if len(unplayed_arms) > 0:  #G
            arm = unplayed_arms[0]  #G
        else:
            ucb_values = self.q_values + self.c * np.sqrt(np.log(self.total_plays) / self.n_values)  #H
            arm = np.argmax(ucb_values)  #H
        return arm

    def update(self, chosen_arm, reward):  #I
        self.total_plays += 1
        self.n_values[chosen_arm] += 1
        self.q_values[chosen_arm] += (reward - self.q_values[chosen_arm]) / self.n_values[chosen_arm]  #J

# Callout annotations (from the book):
#   #A Implements the UCB (Upper Confidence Bound) algorithm.
#   #B Initializes the UCB algorithm parameters. num_arms - The total number of arms (actions) available, and c - The exploration parameter (controls the trade-off between exploration and exploitation).
#   #C Estimated value (Q-value) for each arm
#   #D Count of times each arm has been selected
#   #E Total number of steps/plays in the simulation
#   #F Selects an arm using the UCB rule
#   #G Find all unplayed arms
#   #H Calculate UCB values for all arms that have been played at least once, the uncertainty component increases over time for arms that are not selected, due to total_plays increasing and n_values staying the same for unselected arms.
#   #I Updates the estimated value and count for the chosen arm after receiving a reward. chosen_arm - index of the selected arm and reward received from playing the chosen arm.
#   #J Incremental update rule for Q-value
