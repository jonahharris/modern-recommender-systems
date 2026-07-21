# Figure - Listing 7.25: Pairing descriptions with semantic IDs
# Source: chapters/ch09.md lines 1157-1178
# Chapter: 9
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class AnchoredFormatter(BaseFormatter):
  def __init__(self, item_df):
    super().__init__(item_df)
    self.item_descriptions = dict(
      zip(item_df['id'], item_df['genre_tags'])  #A
    )

  def format(self, user_record, is_training=True):
    tokens = []

    preferences = user_record.get('stated_preferences', [])
    for pref in preferences:
      tokens.append(f"[LIKES_{pref.upper()}]")  #B

    for item_uuid in user_record['history']:
      description = self.item_descriptions.get(
        item_uuid, "UNKNOWN"
      )
      tokens.append(f"[{description}]")  #C
      tokens.extend(self.get_item_tokens(item_uuid))  #D

    return " ".join(tokens)

# Callout annotations (from the book):
#   #A Map item UUIDs to short genre descriptors
#   #B Stated preferences become tokens like \[LIKES\_SCIFI\]
#   #C Natural language anchor before each item
#   #D Semantic IDs follow immediately
