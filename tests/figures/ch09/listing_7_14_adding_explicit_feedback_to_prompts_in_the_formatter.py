# Figure - Listing 7.14: Adding explicit feedback to prompts in the Formatter
# Source: chapters/ch09.md lines 458-472
# Chapter: 9
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class FeedbackFormatter(BaseFormatter):  #A
  def __init__(self, item_df):
    super().__init__(item_df)

  def format(self, user_record, is_training=True):
    tokens = []
    feedback = user_record.get('feedback', {})  #B
    for item_uuid in user_record['history']:
      sentiment = feedback.get(item_uuid, 'VIEWED')  #C
      if sentiment == 'dislike':
        tokens.append("SENTIMENT_NEG")  #D
      elif sentiment == 'like':
        tokens.append("SENTIMENT_POS")
      tokens.extend(self.get_item_tokens(item_uuid))  #E
    return " ".join(tokens)

# Callout annotations (from the book):
#   #A Inherits from BaseFormatter
#   #B Get feedback dictionary, default empty
#   #C Default to 'VIEWED' if no explicit signal
#   #D Prepend sentiment token
#   #E Semantic IDs follow immediately
