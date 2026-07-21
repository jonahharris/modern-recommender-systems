# Figure — Listing 15.1: A/B experiment
# Source: chapters/ch13.md lines 78-81
# Chapter: 13
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A 50% of the time
#   #B call the control implementation
#   #C call the treatment implementation
if random.uniform(0,1) < 0.5:
  return control()
else:
  return variant()
