# Figure — Listing 10.10: Agent tools for movie recommendation
# Source: chapters/ch10.md lines 414-442
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Single retrieval tool (both content and collaborative filtering)
#   #B Filtering remains separate since it operates on already-retrieved results
tools = [
  {
    "name": "search_movies",
    "description": "Search for movies by description "
      "or similarity. Optionally personalize results "
      "using collaborative filtering if a user_id is "
      "provided.",
    "parameters": {
      "query": "str: natural language search query",
      "user_id": "int: user ID for personalization "
                 "(optional)",
      "k": "int: number of results (default 10)"
    },
    "function": retriever.search
  },
  {
    "name": "filter_movies",
    "description": "Filter a list of movies by genre, "
      "year range, or minimum rating. Use after search "
      "to narrow results.",
    "parameters": {
      "movies": "list: movies to filter",
      "genre": "str: genre to filter by (optional)",
      "year_min": "int: minimum year (optional)",
      "year_max": "int: maximum year (optional)"
    },
    "function": filter_by_metadata
  }
]
