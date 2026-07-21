# Figure - Listing 2.21: Three-signal ranking
# Source: chapters/ch02.md lines 855-868
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def rank_three_signals(candidates, content_weight=0.3, behavioral_weight=0.5, k=10):
  popularity_weight = 1.0 - content_weight - behavioral_weight  #A

  for item in candidates:  #B
    item['final_score'] = (
      content_weight * item.get('content_similarity', 0) +
      behavioral_weight * item.get('behavioral_similarity', 0) +
      popularity_weight * item.get('popularity', 0)
    )

  ranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
  return ranked[:k]

# Callout annotations (from the book):
#   #A Calculate weights (must sum to 1.0)
#   #B Weighted combination of all three signals
