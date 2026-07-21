# Figure - Listing 10.3: Wrapping existing recommendations with LLM explanations
# Source: chapters/ch10.md lines 140-157
# Chapter: 10
# Category: api-drift  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def explain_recommendations(llm, user_profile,
                             recommendations):  #A
  items_text = "\n".join(
    f"- {r['title']} ({r.get('year', '')}) "
    f"[{r.get('genres', '')}]"
    for r in recommendations
  )
  return llm.generate(
    system_prompt="You are a movie recommendation "
      "assistant. The user has been recommended these "
      "movies by our system. Explain why each one "
      "might appeal to them based on their profile. "
      "Be concise and specific.",
    user_message=f"User profile: {user_profile}\n\n"
      f"Recommendations:\n{items_text}"
  )  #B
#A Add explanations to an existing ranked list.
#B Call the LLM with a prompt containing recs and user profile.
