# Figure — Listing 15.9: UCB bandit
# Source: chapters/ch13.md lines 509-547
# Chapter: 13
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Implements a Bayesian bandit using Thompson Sampling.
#   #B Initializes the UCB algorithm parameters. num\_arms \- The total number of arms (actions) available, and c \- The exploration parameter (controls the trade-off between exploration and exploitation).
#   #C Estimated value (Q-value) for each arm
#   #D Count of times each arm has been selected
#   #E Total number of steps/plays in the simulation
#   #G Find all unplayed arms
#   #H Calculate UCB values for all arms that have been played at least once, the uncertainty component increases over time for arms that are not selected, due to total\_plays increasing and n\_values staying the same for unselected arms.
#   #I Updates the estimated value and count for the chosen arm after receiving a reward. Chosen\_arm \- index of the selected arm and reward received from playing the chosen arm.
#   #J Incremental update rule for Q-value
class UCBAlgorithm:

  def __init__(self, num_arms, c=1.0):

    self.num_arms = num_arms

    self.c = c

    self.q_values = np.zeros(num_arms)

    self.n_values = np.zeros(num_arms)

    self.total_plays = 0

  def select_arm(self):

    unplayed_arms = np.where(self.n_values == 0)[0]

    if len(unplayed_arms) > 0:

      arm = unplayed_arms[0]

    else:

      ucb_values = self.q_values + self.c * np.sqrt(np.log(self.total_plays) / self.n_values)

      arm = np.argmax(ucb_values)

  return arm

  def update(self, chosen_arm, reward):

    self.total_plays += 1

    self.n_values[chosen_arm] += 1

    self.q_values[chosen_arm] += (reward - self.q_values[chosen_arm]) /

                                  Self.n_values[chosen_arm]
