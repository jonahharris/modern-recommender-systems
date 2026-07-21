# Figure — Listing 5.5: Spot check, query the index with Toy Story's embedding
# Source: chapters/ch05.md lines 246-249
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
toy_story_idx = movie_to_idx[toy_story_id]
query = item_embeddings[toy_story_idx]
ids, scores = query_index(hnsw_index, query, k=5)
print("Toy Story neighbors:", [idx_to_title[i] for i in ids])
