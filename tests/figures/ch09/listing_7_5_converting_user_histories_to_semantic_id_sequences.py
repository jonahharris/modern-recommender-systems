# Figure - Listing 7.5: Converting user histories to semantic ID sequences
# Source: chapters/ch09.md lines 173-201
# Chapter: 9
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class BaseFormatter:
  def __init__(self, item_df):
    self.item_map = dict(zip(item_df['id'], item_df['final_id']))  #A
    self.tokens_missing = Counter()

  def get_item_tokens(self, item_uuid):  #B
    if item_uuid not in self.item_map:
      return []
    s = self.item_map[item_uuid]
    return [
      f"L1_{s[0]}",
      f"L2_{s[1]}",
      f"L3_{s[2]}",
      f"LF_{s[3]}"
    ]

  def format(self, user_record, is_training=True):  #C
    tokens = []
    for t in user_record['history']:  #D
      if t in self.item_map:
        tokens.extend(self.get_item_tokens(t))
      else:
        self.tokens_missing.update([t])  #E
    return " ".join(tokens)
#A UUID → semantic ID tuple lookup
#B Convert one item UUID to four tokens
#C Format an entire user history
#D Iterate over interaction history
#E Track unmapped items for debugging
