# Figure - Listing 5.6: Hard negative mining
# Source: chapters/ch05.md lines 357-382
# Chapter: 5
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def hard_negative_mining(
  user_embeddings: torch.Tensor,
  item_embeddings: torch.Tensor,
  positive_indices: torch.Tensor,
  num_hard_negatives: int = 4,
  num_random_negatives: int = 4,
) -> torch.Tensor:
  similarities = torch.matmul(
    user_embeddings, item_embeddings.T
  )  #A
  batch_size = user_embeddings.shape[0]
  for i in range(batch_size):
    similarities[i, positive_indices[i]] = -float("inf")  #B

  _, hard_neg_indices = torch.topk(
    similarities, k=num_hard_negatives, dim=1
  )  #C

  num_items = item_embeddings.shape[0]
  random_neg_indices = torch.randint(
    0, num_items, (batch_size, num_random_negatives)
  )  #D

  return torch.cat(
    [hard_neg_indices, random_neg_indices], dim=1
  )  #E

# Callout annotations (from the book):
#   #A Compute similarity between all user and item embeddings
#   #B Mask positive items so they cannot be selected as negatives
#   #C Select the highest-scoring non-positive items as hard negatives
#   #D Sample random negatives for training stability
#   #E Combine hard and random negatives
