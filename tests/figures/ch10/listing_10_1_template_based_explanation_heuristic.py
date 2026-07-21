# Figure — Listing 10.1: Template-based explanation heuristic
# Source: chapters/ch10.md lines 84-90
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Looks up what genres the user has watched most
#   #B Retrieves the item's genre tags
#   #C Finds the intersection
#   #D Generic fallback when no genre overlap exists
def generate_explanation_heuristic(user_id, item_id):
  user_genres = get_user_favorite_genres(user_id)
  item_genres = get_item_genres(item_id)
  overlap = set(user_genres) & set(item_genres)
  if overlap:
    return f"Because you like {', '.join(overlap)} movies"
  return "Popular with similar users"
