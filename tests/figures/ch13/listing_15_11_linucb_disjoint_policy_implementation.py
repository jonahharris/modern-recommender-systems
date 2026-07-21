# Figure - Listing 15.11: linucb disjoint policy implementation
# Source: chapters/ch13.md lines 693-727
# Chapter: 13
# Category: needs-prior  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
class linucb_policy():  #A

    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index=i,
                                                d=d,
                                                alpha=alpha) for i in range(K_arms)]  #B

    def select_arm(self, x_array):  #C
        highest_ucb = -1  #D
        candidate_arms = []  #E
        for arm_index in range(self.K_arms):  #F
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)  #G
            if arm_ucb > highest_ucb:  #H
                highest_ucb = arm_ucb  #I
                candidate_arms = [arm_index]  #J
            elif arm_ucb == highest_ucb:  #K
                candidate_arms.append(arm_index)  #K
        chosen_arm = np.random.choice(candidate_arms)  #K
        return chosen_arm

# Callout annotations (from the book):
#   #A Implements a linUcb policy class.
#   #B Initialize all arms, in this version we don’t add any features to the arms
#   #C select_arm method
#   #D Initiate ucb to be -1
#   #E Track index of arms to be selected, if there are more than one arm with the max UCB.
#   #F Calculate ucb based on each arm using current covariates at time t
#   #G call the calc_ucb for each arm
#   #H If current arm is higher than current highest_ucb
#   #I Set new max ucb
#   #J Reset candidate_arms list with new entry based on current arm
#   #K If there is a tie, append to candidate_arms, Choose based on candidate_arms randomly (tie breaker)
