# Figure - Listing 7.11: Example user profile data
# Source: chapters/ch09.md lines 374-378
# Chapter: 9
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
rich_data = [
  {"uid": 1, "age": 25, "loc": "US", "history": ["uuid_101", "uuid_102"]},
  {"uid": 2, "age": 10, "loc": "UK", "history": ["uuid_050", "uuid_051"]},
  {"uid": 3, "history": ["uuid_105", "uuid_101"]}  #A
]

# Callout annotations (from the book):
#   #A No age or location available
