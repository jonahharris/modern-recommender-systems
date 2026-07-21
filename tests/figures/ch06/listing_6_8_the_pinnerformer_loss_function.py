# Figure - Listing 6.8: The Pinnerformer loss function
# Source: chapters/ch06.md lines 550-564
# Chapter: 6
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def pinnerformer_loss(user_emb, future_items, neg_items, num_items, t=0.75):
  n_neg = neg_items.shape[-1]
  alpha = n_neg / max(num_items - 1, 1)
  beta = alpha * (t * (1.0 - 1.0 / alpha) + 1.0 / alpha)
  user_exp = user_emb.unsqueeze(1).expand_as(  #A
      torch.zeros(user_emb.shape[0],
      future_items.shape[1], user_emb.shape[-1])
  )  #B
  pos_scores = (user_exp * item_emb(future_items)).sum(-1)  #C
  neg_scores = (user_exp.unsqueeze(2) * item_emb(neg_items)).sum(-1)  #D

  pos_loss = beta * F.softplus(-pos_scores)
  neg_loss = F.softplus(neg_scores).sum(dim=-1)
  per_pos  = (pos_loss + neg_loss) / (n_neg + 1)
  return per_pos.mean()

# Callout annotations (from the book):
#   #A Expand the single user embedding to score against each future position.
#   #B The same user snapshot competes against every item in the future window.
#   #C Calculate the **positive** score using (batch, window)
#   #D Calculate the **negative** score using (batch, window, n\_neg)
