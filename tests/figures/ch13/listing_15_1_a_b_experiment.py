# Figure - Listing 15.1: A/B experiment
# Source: chapters/ch13.md lines 78-81
# Chapter: 13
# Category: pseudocode  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
if random.uniform(0,1) < 0.5:  #A
  return control()  #B
else:
  return variant()  #C

# Callout annotations (from the book):
#   #A 50% of the time
#   #B call the control implementation
#   #C call the treatment implementation
