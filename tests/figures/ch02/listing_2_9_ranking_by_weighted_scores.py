# Figure - Listing 2.9: Ranking by weighted scores
# Source: chapters/ch02.md lines 313-326
# Chapter: 2
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def rank_candidates(candidates, similarity_weight=0.7, k=10):
    for item in candidates:
        item['final_score'] = (
            similarity_weight * item['similarity'] +
            (1 - similarity_weight) * item['popularity']
        )  #A

    ranked = sorted(
        candidates,
        key=lambda x: x['final_score'],
        reverse=True
    )  #B

    return ranked[:k]  #C

# Callout annotations (from the book):
#   #A Compute weighted combination of scores
#   #B Sort by final score descending
#   #C Return top k
