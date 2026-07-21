# Figure — Listing 7.14: Adding explicit feedback to prompts in the Formatter
# Source: chapters/ch09.md lines 458-472
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Inherits from BaseFormatter
#   #B Get feedback dictionary, default empty
#   #C Default to 'VIEWED' if no explicit signal
#   #D Prepend sentiment token
#   #E Semantic IDs follow immediately
class FeedbackFormatter(BaseFormatter):
  def __init__(self, item_df):
    super().__init__(item_df)

  def format(self, user_record, is_training=True):
    tokens = []
    feedback = user_record.get('feedback', {})
    for item_uuid in user_record['history']:
      sentiment = feedback.get(item_uuid, 'VIEWED')
      if sentiment == 'dislike':
        tokens.append("SENTIMENT_NEG")
      elif sentiment == 'like':
        tokens.append("SENTIMENT_POS")
      tokens.extend(self.get_item_tokens(item_uuid))
    return " ".join(tokens)
