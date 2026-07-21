# Figure - Listing 15.8: An abstract bandit
# Source: chapters/ch13.md lines 456-470
# Chapter: 13
# Category: pseudocode  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
class Bandit:  #A

    def select_arm(self, context):  #B
        arm_values = []
        for arm in context.available_arms:  #C
            arm_values.append(arm.calc_value(context))  #D
        return np.argmax(arm_values)  #E

    def update_estimates(self, context, chosen_arm, reward):  #F
        chosen_arm.update(context, reward)  #G

# Callout annotations (from the book):
#   #A General pattern for bandits
#   #B All bandits have a select_arm method, which takes a context
#   #C Iterate over the arms, note that the context dictates which arms are available
#   #D Calculate the arm value, might be based on a context
#   #E Select the arm with highest value
#   #F All bandits also have an update method, where they learn from the feedback
#   #G Update is usually just delegated to the chosen_arm by calling update on it
