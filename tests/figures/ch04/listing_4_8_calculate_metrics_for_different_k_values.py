# Figure - Listing 4.8: Calculate metrics for different K values
# Source: chapters/ch04.md lines 509-542
# Chapter: 4
# Category: needs-package  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
for k in k_values:  #A
  if k <= len(top_k_indices):  #B
    top_k_rel = relevance_scores[:k]  #C

    if np.sum(top_k_rel) > 0:  #D
      rank_scores = np.arange(len(top_k_rel), 0, -1)  #E
      ndcg_k = ndcg_score([top_k_rel], [rank_scores])
      all_metrics[k]['ndcg'].append(ndcg_k)

    precision_k = np.sum(top_k_rel) / k  #F
    all_metrics[k]['precision'].append(precision_k)

    if len(test_indices) > 0:
      recall_k = np.sum(top_k_rel) / len(test_indices)  #G
      all_metrics[k]['recall'].append(recall_k)

    first_relevant = np.where(top_k_rel == 1)[0]  #H
    if len(first_relevant) > 0:
      mrr = 1.0 / (first_relevant[0] + 1)
      all_metrics[k]['mrr'].append(mrr)
    else:
      all_metrics[k]['mrr'].append(0.0)

    if np.sum(top_k_rel) > 0:  #I
      ap = 0.0
      relevant_found = 0
      for i, rel in enumerate(top_k_rel):
        if rel == 1:
          relevant_found += 1
          ap += relevant_found / (i + 1)
      ap /= min(len(test_indices), k)
      all_metrics[k]['map'].append(ap)
    else:
      all_metrics[k]['map'].append(0.0)

# Callout annotations (from the book):
#   #A For each k, calculate the metrics
#   #B Only if there are enough recommended items
#   #C Look at only the top k items
#   #D NDCG@K (only calculate if there are relevant items)
#   #E Descending position scores define the ranking order
#   #F Precision@K
#   #G Recall@K
#   #H Mean Reciprocal Rank (MRR)
#   #I Mean Average Precision (MAP)
