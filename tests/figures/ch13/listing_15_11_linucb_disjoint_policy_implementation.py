# Figure — Listing 15.11: linucb disjoint policy implementation
# Source: chapters/ch13.md lines 693-727
# Chapter: 13
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Implements a linUcb policy class.
#   #B Initialize all arms, in this version we don’t add any features to the arms
#   #C select\_arm method
#   #D Initiate ucb to be \-1
#   #E Track index of arms to be selected, if there are more than one arm with the max UCB.
#   #F Calculate ucb based on each arm using current covariates at time t
#   #G call the calc\_ucb for each arm
#   #H If current arm is highest than current highest\_ucb
#   #I Set new max ucb
#   #J Reset candidate\_arms list with new entry based on current arm
#   #K If there is a tie, append to candidate\_arms, Choose based on candidate\_arms randomly (tie breaker)
class linucb_policy():

    def __init__(self, K_arms, d, alpha):

        self.K_arms = K_arms

        self.linucb_arms = [linucb_disjoint_arm(arm_index = 1,

                                                d = d,

                                                alpha = alpha) for i in range(K_arms)]

  def select_arm(self, x_array):

    highest_ucb = -1

    candidate_arms = []

    for arm_index in range(self.K_arms):

      arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)

      if arm_ucb > highest_ucb:

        highest_ucb = arm_ucb

        candidate_arms = [arm_index]

        if arm_ucb == highest_ucb:

          candidate_arms.append(arm_index)

    chosen_arm = np.random.choice(candidate_arms)

    return chosen_arm
