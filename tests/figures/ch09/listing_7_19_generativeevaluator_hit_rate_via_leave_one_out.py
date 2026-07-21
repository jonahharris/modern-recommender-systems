# Figure - Listing 7.19: GenerativeEvaluator: hit rate via leave-one-out
# Source: chapters/ch09.md lines 779-834
# Chapter: 9
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class GenerativeEvaluator:
  def __init__(self, model, tokenizer, formatter, k=10):
    self.model = model
    self.tokenizer = tokenizer
    self.formatter = formatter
    self.k = k

  def evaluate_user(self, user_data):
    full_history = user_data['history']
    if len(full_history) < 2:
      return 0, 0

    train_history = full_history[-11:-1]  #A
    target_uuid = full_history[-1]  #B

    target_tokens = self.formatter.get_item_tokens(target_uuid)
    if not target_tokens:
      return 0, 0
    target_string = " ".join(target_tokens)

    known_items = [t for t in train_history
                   if t in self.formatter.item_map]
    if len(known_items) == 0:
      return 0, 0

    if target_uuid not in self.formatter.item_map:
      return 0, 0

    eval_user = user_data.copy()
    eval_user['history'] = train_history
    recs = generate_recommendations(  #C
      self.model, self.tokenizer, self.formatter,
      eval_user, num_items=self.k
    )

    hit = 0
    ndcg = 0
    if target_string in recs:
      hit = 1
      rank = recs.index(target_string)  #D
      ndcg = 1.0 / np.log2(rank + 2)  #E

    return hit, ndcg

  def run_benchmark(self, test_data, limit=100):
    total_hr = []
    total_ndcg = []
    for i in range(min(limit, len(test_data))):
      hr, ndcg = self.evaluate_user(test_data.iloc[i])
      total_hr.append(hr)
      total_ndcg.append(ndcg)
    return {
      "k": self.k,
      "hit_rate": np.mean(total_hr),
      "NDCG": np.mean(total_ndcg),
    }

# Callout annotations (from the book):
#   #A Last 10 items as context
#   #B The held-out item we want the model to predict
#   #C Generate K recommendations
#   #D Find where the target appears in the list
#   #E NDCG: position 0 → 1.0, position 1 → 0.63, etc.
