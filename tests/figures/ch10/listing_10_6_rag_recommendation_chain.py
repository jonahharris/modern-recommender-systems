# Figure — Listing 10.6: RAG recommendation chain
# Source: chapters/ch10.md lines 300-319
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Format each candidate with its metadata so the LLM can reason about it
#   #B Optional user profile adds personalization context
#   #C Last three exchanges (6 turns) provide conversational context without exceeding the context window
def build_recommendation_prompt(query, candidates,
                                user_profile=None,
                                conversation_history=None):

  candidates_text = "\n".join(
    f"- {c['title']} ({c.get('year', 'N/A')}) "
    f"[{c.get('genres', '')}]: {c.get('overview', '')}"
    for c in candidates
    )
  user_message = f"Request: {query}\n\n"
  user_message += f"Candidates:\n{candidates_text}\n"
  if user_profile:
    user_message += f"\nUser profile: {user_profile}\n"
  if conversation_history:
    history_text = "\n".join(
      f"{turn['role']}: {turn['content']}"
      for turn in conversation_history[-6:]
      )
    user_message += f"\nConversation so far:\n{history_text}\n"
  return system_prompt, user_message
