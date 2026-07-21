# Figure — Listing 7.24: ICL reranking over a candidate set
# Source: chapters/ch09.md lines 1107-1125
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Use the last 5 items for recency
#   #B Format candidates as a numbered list
#   #C Call the LLM API (e.g., GPT-4, Claude)
def icl_rerank(candidates, user_history,
               user_context, llm):
  history_str = ", ".join(user_history[-5:])
  candidates_str = "\n".join(
    [f"{i+1}. {c}" for i, c in enumerate(candidates)]
  )

  prompt = f"""You are a personalized recommender.

User's recent watch history: {history_str}
Context: {user_context}

From the following candidates, rank the top 3
most relevant:
{candidates_str}

Top 3 recommendations:"""

  return llm.generate(prompt)
