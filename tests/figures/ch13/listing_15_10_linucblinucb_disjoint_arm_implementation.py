# Figure - Listing 15.10: LinUCBlinucb disjoint arm implementation
# Source: chapters/ch13.md lines 635-665
# Chapter: 13
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
class linucb_disjoint_arm():  #A

    def __init__(self, arm_index, d, alpha):
        self.arm_index = arm_index  #B
        self.alpha = alpha  #C
        self.A = np.identity(d)  #D
        self.b = np.zeros([d, 1])  #E

    def calc_UCB(self, x_array):  #F
        A_inv = np.linalg.inv(self.A)  #G
        self.theta = np.dot(A_inv, self.b)  #H
        x = x_array.reshape([-1, 1])  #I
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))  #J
        return p

    def reward_update(self, reward, x_array):  #K
        x = x_array.reshape([-1, 1])  #L
        self.b += reward * x  #L

# Callout annotations (from the book):
#   #A Implements a linUcb arm class.
#   #B Track arm index
#   #C alpha is the confidence bound, meaning that you adjust this to balance exploitation and exploration
#   #D The inverse of A is used in ridge regression
#   #E corresponding response vector.
#   #F method which calculates the UCB.
#   #G Find A inverse for ridge regression
#   #H Perform ridge regression to obtain estimate of covariate, coefficients theta
#   #I Reshape covariates input into (d x 1) shape vector
#   #J Find the ucb-value (expected_value + uncertainty)
#   #K Method which updates the arm from the observed reward
#   #L Update the weights of the linear function
