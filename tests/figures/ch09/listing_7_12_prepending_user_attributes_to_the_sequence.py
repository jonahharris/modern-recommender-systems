# Figure - Listing 7.12: Prepending user attributes to the sequence
# Source: chapters/ch09.md lines 384-413
# Chapter: 9
# Category: needs-prior  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
import random

class ContextualFormatter(BaseFormatter):  #A
  def __init__(self, item_df, dropout_rate=0.1):
    super().__init__(item_df)
    self.dropout_rate = dropout_rate

  def format(self, user_record, is_training=True):
    tokens = []

    val_age = user_record.get('age')  #B
    if val_age is not None and not pd.isna(val_age):
      if is_training and random.random() < self.dropout_rate:  #C
        pass
      else:
        tokens.append(f"AGE_{int(val_age)}")
    else:
      tokens.append("AGE_UNK")  #D

    val_loc = user_record.get('loc')  #E
    if val_loc is not None and not pd.isna(val_loc):
      if is_training and random.random() < self.dropout_rate:
        pass
      else:
        tokens.append(f"LOC_{val_loc}")
    else:
      tokens.append("LOC_UNK")

    for item_uuid in user_record['history']:  #F
      tokens.extend(self.get_item_tokens(item_uuid))

    return " ".join(tokens)

# Callout annotations (from the book):
#   #A Inherits from BaseFormatter
#   #B Check for age data
#   #C Token dropout (10%)
#   #D Explicit unknown marker
#   #E Same pattern for location
#   #F Append semantic IDs for history items
