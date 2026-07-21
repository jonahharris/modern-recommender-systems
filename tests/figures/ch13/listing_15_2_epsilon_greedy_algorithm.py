# Figure — Listing 15.2: Epsilon-greedy algorithm
# Source: chapters/ch13.md lines 146-152
# Chapter: 13
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A if random number is less than epsilon then explore
#   #B select an arm at random
#   #C otherwise select the arm with the highest estimated value
if np.random.random() < epsilon:

  return np.random.randint(num_arms)

else:

  return np.argmax(estimated_arm_values)
