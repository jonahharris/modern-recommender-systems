# Figure - Listing 10.16: Adding a tool to the agent toolbox
# Source: chapters/ch10.md lines 678-690
# Chapter: 10
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
tools.append({
  "name": "generate_sequential",
  "description": "Generate recommendations based on "
    "a user's watch history sequence. Best for "
    "continuing a viewing pattern or finding what "
    "naturally follows from recent watches. Requires "
    "a user record with history.",
  "parameters": {
    "user_record": "dict: user record with history",
    "num_items": "int: number of items (default 10)"
  },
  "function": gen_tool.recommend_from_history
})
