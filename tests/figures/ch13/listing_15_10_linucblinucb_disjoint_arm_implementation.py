# Figure — Listing 15.10: LinUCBlinucb disjoint arm implementation
# Source: chapters/ch13.md lines 635-665
# Chapter: 13
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Implements a linUcb arm class.
#   #B Track arm index
#   #C alpha is the confidence bound, meaning that you adjust this to balance exploitation and exploration
#   #D The inverse of A is used in ridge regression
#   #E corresponding response vector.
#   #F method which calculates the UCB.
#   #G Find A inverse for ridge regression
#   #H Perform ridge regression to obtain estimate of covariate, coefficients theta
#   #I Reshape covariates input into (d x 1\) shape vector
#   #J Find the ucb-value (expected\_value \+ uncertainty)
#   #L Update the weights of the linear function
class linucb_disjoint_arm():

   def __init__(self, arm_index, d, alpha):

     self.arm_index = arm_index

     self.alpha = alpha

     self.A = np.identity(d)

     self.b = np.zeros([d,1])

  def calc_UCB(self, x_array):

    A_inv = np.linalg.inv(self.A)

    self.theta = np.dot(A_inv, self.b)



       x = x_array.reshape([-1,1])

              p = np.dot(self.theta.T,x) +  self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv,x)))

       return p

  def reward_update(self, reward, x_array):

    x = x_array.reshape([-1,1])

    self.b += reward * x
