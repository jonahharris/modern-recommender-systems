# Figure — Listing 7.12: Prepending user attributes to the sequence
# Source: chapters/ch09.md lines 384-413
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Inherits from BaseFormatter
#   #B Check for age data
#   #C Token dropout (10%)
#   #D Explicit unknown marker
#   #E Same pattern for location
#   #F Append semantic IDs for history items
class ContextualFormatter(BaseFormatter):
  def __init__(self, item_df, dropout_rate=0.1):
    super().__init__(item_df)
    self.dropout_rate = dropout_rate

  def format(self, user_record, is_training=True):
    tokens = []

    val_age = user_record.get('age')
    if val_age is not None and not pd.isna(val_age):
      if is_training and random.random() < self.dropout_rate:
        pass
      else:
        tokens.append(f"AGE_{int(val_age)}")
    else:
      tokens.append("AGE_UNK")

    val_loc = user_record.get('loc')
    if val_loc is not None and not pd.isna(val_loc):
      if is_training and random.random() < self.dropout_rate:
        pass
      else:
        tokens.append(f"LOC_{val_loc}")
    else:
      tokens.append("LOC_UNK")

    for item_uuid in user_record['history']:
      tokens.extend(self.get_item_tokens(item_uuid))

    return " ".join(tokens)
