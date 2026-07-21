# Figure - Listing 10.1: Template-based explanation heuristic
# Source: chapters/ch10.md lines 84-90
# Chapter: 10
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def generate_explanation_heuristic(user_id, item_id):
  user_genres = get_user_favorite_genres(user_id)  #A
  item_genres = get_item_genres(item_id)  #B
  overlap = set(user_genres) & set(item_genres)  #C
  if overlap:
    return f"Because you like {', '.join(overlap)} movies"
  return "Popular with similar users"  #D

# Callout annotations (from the book):
#   #A Looks up what genres the user has watched most
#   #B Retrieves the item's genre tags
#   #C Finds the intersection
#   #D Generic fallback when no genre overlap exists
