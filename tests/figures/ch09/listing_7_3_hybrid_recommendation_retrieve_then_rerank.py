# Figure — Listing 7.3: Hybrid recommendation: retrieve then rerank
# Source: chapters/ch09.md lines 108-113
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
def recommend_hybrid(user_history, catalog):
  candidates = traditional_recsys.get_candidates(user_history, k=100)
  final_recs = llm.rerank_and_explain(candidates, k=10)
  return final_recs
#A Traditional system generates candidates (fast, catalog-safe)
#B Generative system reranks and optionally explains (flexible, high-quality)
