# Figure — Listing 7.2: Generative recommendation: prompt and generate
# Source: chapters/ch09.md lines 51-56
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
prompt = f"User has watched: {user_history}. Recommend 5 similar movies."
recommendations = llm.generate(prompt)
return recommendations

#A Create a prompt string from user history
#B Request recommendations from the LLM
