# Figure - Listing 1.4: Calculating similarity between comics
# Source: chapters/ch01.md lines 199-208
# Chapter: 1
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def content_similarity(comic1_id, comic2_id):
  tags1 = comic_tags[comic1_id]
  tags2 = comic_tags[comic2_id]
  intersection = len(tags1 & tags2)  #A
  union = len(tags1 | tags2)  #B
  return intersection/union if union > 0 else 0  #C

#A Count tags that appear in both comics
#B Count all unique tags across both comics
#C Jaccard similarity: shared tags divided by total unique tags
