# Figure — Listing 7.4: Converting an item's semantic ID to prefixed tokens
# Source: chapters/ch09.md lines 153-165
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
def get_item_tokens(self, item_uuid):
  if item_uuid not in self.item_map:
    return []
  s = self.item_map[item_uuid]
  return [
    f"L1_{s[0]}",
    f"L2_{s[1]}",
    f"L3_{s[2]}",
    f"LF_{s[3]}"
  ]
#A Return empty if item has no semantic ID
#B Level prefix (L1, L2, L3)
#C Leaf node prefix
