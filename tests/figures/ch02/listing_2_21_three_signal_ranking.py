# Figure — Listing 2.21: Three-signal ranking
# Source: chapters/ch02.md lines 855-868
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Calculate weights (must sum to 1.0)
#   #B Weighted combination of all three signals
#   #C Display all scores
def rank_three_signals(candidates, content_weight=0.3, behavioral_weight=0.5, k=10):
  popularity_weight = 1.0 - content_weight - behavioral_weight

  ranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
  return ranked[:k]

  seed_movie_id = 1

  for item in candidates:
    item['final_score'] = (
      content_weight * item.get('content_similarity', 0) +
      behavioral_weight * item.get('behavioral_similarity', 0) +
      popularity_weight * item.get('popularity', 0)
    )
