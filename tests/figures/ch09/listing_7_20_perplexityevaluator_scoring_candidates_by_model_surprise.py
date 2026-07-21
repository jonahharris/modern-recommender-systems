# Figure — Listing 7.20: PerplexityEvaluator: scoring candidates by model surprise
# Source: chapters/ch09.md lines 856-949
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Concatenate history, candidate, and end-of-text marker
#   #B Find where the candidate tokens start
#   #C Shift logits and labels for next-token prediction
#   #D Slice to candidate tokens only
#   #E Sum, not mean (see explanation below)
#   #F Negative loss: lower loss \= higher score
#   #G Everything except the last item is context
#   #H Sample 99 negative items
#   #I True target is always at index 0
#   #J Sort by score, highest first
#   #K Check if target (index 0\) is in top K
#   #L NDCG based on position
class PerplexityEvaluator:
  def __init__(self, model, tokenizer, formatter,
               all_items_set, k=10):
    self.model = model
    self.tokenizer = tokenizer
    self.formatter = formatter
    self.all_items = list(all_items_set)
    self.k = k
    self.device = model.device

  def calculate_score(self, history_str, candidate_str):
    full_text = (history_str + " " + candidate_str
                 + " <|endoftext|>")
    inputs = self.tokenizer(
      full_text, return_tensors="pt"
    ).to(self.device)
    input_ids = inputs.input_ids

    history_ids = self.tokenizer.encode(
      history_str, add_special_tokens=False
    )
    start_idx = len(history_ids)

    with torch.no_grad():
      outputs = self.model(input_ids, labels=input_ids)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    target_logits = shift_logits[:, start_idx:, :]
    target_labels = shift_labels[:, start_idx:]

    loss = F.cross_entropy(
      target_logits.transpose(1, 2),
      target_labels,
      reduction='sum'
    )
    return -loss.item()

  def evaluate_user(self, user_data):
    full_history = user_data['history']
    if len(full_history) < 2:
      return 0, 0

    train_history = full_history[:-1]
    target_item = full_history[-1]

    eval_user = user_data.copy()
    eval_user['history'] = train_history
    history_str = self.formatter.format(
      eval_user, is_training=False
    )

    negatives = []
    hist_set = set(full_history)
    while len(negatives) < 99:
      item = np.random.choice(self.all_items)
      if item not in hist_set:
        negatives.append(item)

    candidates = [target_item] + negatives
    scores = []
    for item_uuid in candidates:
      item_tokens = self.formatter.get_item_tokens(item_uuid)
      if not item_tokens:
        scores.append(-9999)
        continue
      item_str = " ".join(item_tokens)
      score = self.calculate_score(history_str, item_str)
      scores.append(score)

    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1]

    hit_rate = 1 if 0 in sorted_indices[:self.k] else 0
    ndcg = 0
    pos = np.where(sorted_indices == 0)[0]
    if len(pos) > 0 and pos[0] < self.k:
      ndcg = 1.0 / np.log2(pos[0] + 2)

    return hit_rate, ndcg

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
