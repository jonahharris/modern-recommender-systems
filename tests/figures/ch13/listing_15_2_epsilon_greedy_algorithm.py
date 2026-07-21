# Figure - Listing 15.2: Epsilon-greedy algorithm
# Source: chapters/ch13.md lines 146-152
# Chapter: 13
# Category: pseudocode  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
if np.random.random() < epsilon:  #A

  return np.random.randint(num_arms)  #B

else:

  return np.argmax(estimated_arm_values)  #C

# Callout annotations (from the book):
#   #A if random number is less than epsilon then explore
#   #B select an arm at random
#   #C otherwise select the arm with the highest estimated value
