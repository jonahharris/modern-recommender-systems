# Figure - Listing 8.17: Semantic coherence tests
# Source: chapters/ch08.md lines 965-973
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
 test_pairs = [
  ("Star Wars", "Star Trek", "high"),  #A
  ("The Notebook",
    "Pride and Prejudice", "high"),  #A
  ("Die Hard", "Lethal Weapon", "high"),  #A
  ("Star Wars", "The Notebook", "low"),  #B
  ("Die Hard", "Frozen", "low"),  #B
  ("Inception", "The Lion King", "low"),  #B
]

# Callout annotations (from the book):
#   #A Should be similar (shared genre)
#   #B Should be dissimilar (different genres)
