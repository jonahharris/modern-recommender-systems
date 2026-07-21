# Figure - Listing 15.8: An abstract bandit
# Source: chapters/ch13.md lines 456-470
# Chapter: 13
# Category: pseudocode  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class Bandit:  #A

  def select_arm(self, context):  #B

    arm_values = []

    For arm in context.availble_arms:  #C

	arm_values.append(arm.calc_value(context))  #D

    return np.argmax(arm_values)  #E

   def update_estimates(self, context, chosen_arm, reward):  #F

	chosen_arm.update(context, reward)  #G

# Callout annotations (from the book):
#   #A General pattern for bandits
#   #B All bandits haves a select\_arm method, which takes a context
#   #C  Iterate over the arms, note that the context dictates which arms are available
#   #D Calculate the arm value, might be based on a context
#   #D Select the arm with highest value
#   #E All bandits also have an update method, where they learn from the feedback.
#   #F Update will usually just be delegated to the chosen\_arm
#   #G calling update on the chosen arm
