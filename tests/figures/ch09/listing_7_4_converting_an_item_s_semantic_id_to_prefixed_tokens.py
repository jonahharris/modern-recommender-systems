# Figure - Listing 7.4: Converting an item's semantic ID to prefixed tokens
# Source: chapters/ch09.md lines 153-165
# Chapter: 9
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def get_item_tokens(self, item_uuid):
  if item_uuid not in self.item_map:  #A
    return []
  s = self.item_map[item_uuid]
  return [
    f"L1_{s[0]}",  #B
    f"L2_{s[1]}",
    f"L3_{s[2]}",
    f"LF_{s[3]}"  #C
  ]
#A Return empty if item has no semantic ID
#B Level prefix (L1, L2, L3)
#C Leaf node prefix
