# Figure — Listing 10.18: Natural-language user taste profiler
# Source: chapters/ch10.md lines 772-793
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Any key-value store — Redis, a database, even a dict for development
#   #B Retrieve the current taste profile.
#   #C Update the profile based on a completed session.
#   #D The LLM synthesizes the existing profile with new signals, resolving contradictions
#   #E Persist the updated profile for the next session
class UserProfiler:
  def __init__(self, llm, profile_store):
    self.llm = llm
    self.store = profile_store

  def get_profile(self, user_id):
    return self.store.get(user_id, "No profile yet.")

  def update_profile(self, user_id,
                     conversation_summary):
    current = self.get_profile(user_id)
    updated = self.llm.generate(
      system_prompt="You maintain user taste profiles "
        "for a movie recommender. Update the profile "
        "below based on the new conversation. Keep it "
        "concise (3-5 sentences). Preserve existing "
        "preferences unless explicitly contradicted.",
      user_message=f"Current profile:\n{current}\n\n"
        f"New conversation:\n{conversation_summary}"
    )
    self.store[user_id] = updated
    return updated
