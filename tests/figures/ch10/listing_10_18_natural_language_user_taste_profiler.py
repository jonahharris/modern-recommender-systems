# Figure - Listing 10.18: Natural-language user taste profiler
# Source: chapters/ch10.md lines 772-793
# Chapter: 10
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class UserProfiler:
  def __init__(self, llm, profile_store):
    self.llm = llm
    self.store = profile_store  #A

  def get_profile(self, user_id):
    return self.store.get(user_id, "No profile yet.")  #B

  def update_profile(self, user_id,
                     conversation_summary):
    current = self.get_profile(user_id)  #C
    updated = self.llm.generate(  #D
      system_prompt="You maintain user taste profiles "
        "for a movie recommender. Update the profile "
        "below based on the new conversation. Keep it "
        "concise (3-5 sentences). Preserve existing "
        "preferences unless explicitly contradicted.",
      user_message=f"Current profile:\n{current}\n\n"
        f"New conversation:\n{conversation_summary}"
    )
    self.store[user_id] = updated  #E
    return updated

# Callout annotations (from the book):
#   #A Any key-value store — Redis, a database, even a dict for development
#   #B Retrieve the current taste profile.
#   #C Update the profile based on a completed session.
#   #D The LLM synthesizes the existing profile with new signals, resolving contradictions
#   #E Persist the updated profile for the next session
