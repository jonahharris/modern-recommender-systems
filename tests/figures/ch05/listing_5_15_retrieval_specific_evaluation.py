# Figure - Listing 5.15: Retrieval-specific evaluation
# Source: chapters/ch05.md lines 906-938
# Chapter: 5
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def evaluate_retrieval(
  retrieval,
  test_users: list,
  relevance_sets: dict[int, set[int]],
  k: int = 100,
) -> dict:
  recalls = []
  all_retrieved = set()

  for user_id in test_users:
    relevant = relevance_sets.get(user_id, set())
    if not relevant:
      continue

    # Use a random relevant item as seed (I2I retrieval)
    seed = next(iter(relevant))
    candidates = retrieval.retrieve_similar_items(seed, k=k)  #A

    candidate_set = set(candidates)
    all_retrieved.update(candidate_set)  #B

    hits = len(candidate_set & relevant)
    recalls.append(hits / len(relevant))  #C

  coverage = len(all_retrieved) / len(
    retrieval.id_to_item
  )  #D

  return {
    "retrieval_recall@k": np.mean(recalls),
    "catalogue_coverage": coverage,
    "num_users_evaluated": len(recalls),
  }

# Callout annotations (from the book):
#   #A Retrieve candidates for this user
#   #B Track all items ever retrieved across users
#   #C Retrieval recall: fraction of relevant items in the candidate pool
#   #D Coverage: fraction of catalog items retrieved for at least one user
