# Figure — Listing 4.8: Calculate metrics for different K values
# Source: chapters/ch04.md lines 509-542
# Chapter: 4
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
for k in k_values:
  if k <= len(top_k_indices):
    top_k_rel = relevance_scores[:k]

    if np.sum(top_k_rel) > 0:
      ideal_rel = np.sort(top_k_rel)[::-1]
      ndcg_k = ndcg_score([ideal_rel], [top_k_rel])
      all_metrics[k]['ndcg'].append(ndcg_k)

    precision_k = np.sum(top_k_rel) / k
    all_metrics[k]['precision'].append(precision_k)

    if len(test_indices) > 0:
      recall_k = np.sum(top_k_rel) / len(test_indices)
      all_metrics[k]['recall'].append(recall_k)

    first_relevant = np.where(top_k_rel == 1)[0]
    if len(first_relevant) > 0:
      mrr = 1.0 / (first_relevant[0] + 1)
      all_metrics[k]['mrr'].append(mrr)
    else:
      all_metrics[k]['mrr'].append(0.0)

    if np.sum(top_k_rel) > 0:
      ap = 0.0
      relevant_found = 0
      for i, rel in enumerate(top_k_rel):
        if rel == 1:
        relevant_found += 1
    ap += relevant_found / (i + 1)
    ap /= np.sum(top_k_rel)
     all_metrics[k]['map'].append(ap)
   else:
     all_metrics[k]['map'].append(0.0)
