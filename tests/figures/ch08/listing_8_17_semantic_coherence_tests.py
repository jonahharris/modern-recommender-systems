# Figure — Listing 8.17: Semantic coherence tests
# Source: chapters/ch08.md lines 965-976
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
 test_pairs = [
  ("Star Wars", "Star Trek", "high"),
  ("The Notebook",
    "Pride and Prejudice", "high"),
  ("Die Hard", "Lethal Weapon", "high"),
  ("Star Wars", "The Notebook", "low"),
  ("Die Hard", "Frozen", "low"),
  ("Inception", "The Lion King", "low"),
]

#A Should be similar (shared genre)
#B Should be dissimilar (different genres)
