# Figure - Listing 7.2: Generative recommendation: prompt and generate
# Source: chapters/ch09.md lines 51-53
# Chapter: 9
# Category: pseudocode  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def recommend_generative(user_history, llm):
  prompt = f"User has watched: {user_history}. Recommend 5 similar movies."  #A
  recommendations = llm.generate(prompt)  #B
  return recommendations

# Callout annotations (from the book):
#   #A Create a prompt string from user history
#   #B Request recommendations from the LLM
