# Figure - Listing 7.17: Mapping generated IDs back to catalog items
# Source: chapters/ch09.md lines 680-700
# Chapter: 9
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class SemanticDecoder:
  def __init__(self, item_df):
    self.id_to_title = dict(
      zip(item_df['final_id'], item_df['title'])
    )  #A

  def parse_tokens(self, token_string):  #B
    try:
      ids = [int(t.split('_')[1]) for t in token_string.split()]
      return tuple(ids)
    except (IndexError, ValueError):
      return None

  def decode(self, token_string):
    semantic_tuple = self.parse_tokens(token_string)  #C
    if not semantic_tuple:
      return "Error: Could not parse tokens"
    if semantic_tuple in self.id_to_title:
      return self.id_to_title[semantic_tuple]  #D
    else:
      return f"Hallucination: {semantic_tuple} not in catalog"  #E

# Callout annotations (from the book):
#   #A Pre-computed reverse index: tuple → title
#   #B Parse "L1\_6 L2\_1 L3\_81 LF\_0" → (6, 1, 81, 0\)
#   #C Convert token string to tuple
#   #D Exact match found
#   #E Generated ID does not exist in catalog
