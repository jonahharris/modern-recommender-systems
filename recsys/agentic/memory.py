"""
Memory components for agentic recommendation:
- ConversationMemory: short-term, within a session
- UserProfiler: long-term, across sessions
- ConversationalRecommender: combines agent + memory
"""

import json
from typing import Dict, List, Optional


class ConversationMemory:
  """Sliding window memory with summarization."""

  def __init__(self, llm, max_recent_turns=6,
               summary_threshold=10):
    self.llm = llm
    self.turns = []
    self.summary = ""
    self.max_recent = max_recent_turns                 #A
    self.threshold = summary_threshold

  def add_turn(self, role, content):
    self.turns.append(
      {"role": role, "content": content}
    )
    if len(self.turns) > self.threshold:               #B
      self._compress()

  def get_context(self):
    """Return summary + recent turns for prompting."""
    context = []
    if self.summary:
      context.append({
        "role": "system",
        "content": f"Summary of earlier conversation: "
                   f"{self.summary}"
      })
    context.extend(self.turns[-self.max_recent:])
    return context

  def clear(self):
    self.turns = []
    self.summary = ""

  def _compress(self):
    """Summarize older turns to stay within budget."""
    old_turns = self.turns[:-self.max_recent]
    turns_text = "\n".join(
      f"{t['role']}: {t['content']}" 
      for t in old_turns
    )
    self.summary = self.llm.generate(                  #C
      system_prompt=(
        "Summarize this conversation history "
        "concisely, preserving: user preferences, "
        "rejected items, and active constraints."
      ),
      user_message=turns_text
    )
    self.turns = self.turns[-self.max_recent:]


class UserProfiler:
  """Natural-language taste profiles persisted across sessions."""

  def __init__(self, llm, profile_store=None):
    self.llm = llm
    self.store = profile_store or {}                   #D

  def get_profile(self, user_id):
    return self.store.get(
      str(user_id), "No profile yet."
    )

  def update_profile(self, user_id, 
                     conversation_summary):
    """Update profile based on a completed session."""
    current = self.get_profile(user_id)
    updated = self.llm.generate(
      system_prompt=(
        "You maintain user taste profiles for a "
        "movie recommender. Update the profile below "
        "based on the new conversation. Keep it "
        "concise (3-5 sentences). Preserve existing "
        "preferences unless explicitly contradicted."
      ),
      user_message=(
        f"Current profile:\n{current}\n\n"
        f"New conversation:\n{conversation_summary}"
      )
    )
    self.store[str(user_id)] = updated                 #E
    return updated

  def save(self, path):
    """Persist profiles to disk."""
    import json
    with open(path, "w") as f:
      json.dump(self.store, f, indent=2)

  def load(self, path):
    """Load profiles from disk."""
    import json
    with open(path, "r") as f:
      self.store = json.load(f)


class ConversationalRecommender:
  """Multi-turn conversational recommender combining 
  agent, memory, and user profiling."""

  def __init__(self, llm, retriever, profiler=None):
    self.llm = llm
    self.retriever = retriever
    self.profiler = profiler
    self.memory = ConversationMemory(llm)
    self.rejected_items = set()                        #F
    self.active_constraints = {}

  def chat(self, user_id, user_message):
    """Handle one turn of conversation."""
    self.memory.add_turn("user", user_message)

    profile = ""
    if self.profiler:
      profile = self.profiler.get_profile(user_id)

    # Retrieve candidates based on the message
    candidates = self.retriever.search(
      user_message, user_id=user_id, k=15
    )

    # Build the prompt
    context = self.memory.get_context()
    system = self._build_system(profile)

    candidates_text = "\n".join(
      f"- {c['title']} ({c.get('year', 'N/A')}) "
      f"[{c.get('genres', '')}]"
      for c in candidates
    )

    augmented_message = (
      f"{user_message}\n\n"
      f"Available candidates from catalog:\n"
      f"{candidates_text}"
    )

    response = self.llm.generate(
      system_prompt=system,
      messages=context[:-1] + [                        #G
        {"role": "user", "content": augmented_message}
      ]
    )

    self._update_constraints(user_message, response)
    self.memory.add_turn("assistant", response)
    return response

  def reset(self):
    """Reset conversation state for a new session."""
    self.memory.clear()
    self.rejected_items.clear()
    self.active_constraints.clear()

  def end_session(self, user_id):
    """End session and update long-term profile."""
    if self.profiler and len(self.memory.turns) > 2:
      summary = "\n".join(
        f"{t['role']}: {t['content']}" 
        for t in self.memory.turns
      )
      self.profiler.update_profile(user_id, summary)
    self.reset()

  def _build_system(self, profile):
    rejected_str = (
      ", ".join(self.rejected_items) 
      if self.rejected_items else "None"
    )
    constraints_str = ""
    if self.active_constraints:
      constraints_str = (
        "\nActive constraints: " +
        ", ".join(
          f"{k}: {v}" 
          for k, v in self.active_constraints.items()
        )
      )

    return f"""You are a movie recommendation assistant 
having a conversation with a user.

User profile: {profile}

Previously rejected this session: {rejected_str}
{constraints_str}

Guidelines:
- If the request is vague, ask ONE targeted question 
  to narrow preferences (mood, genre, or era)
- When recommending, pick from the provided candidates
  and explain why each fits
- When the user critiques, acknowledge what they 
  disliked and adjust
- Never re-recommend a rejected item
- Limit recommendations to 3-5 items per turn
- Be concise but specific in explanations"""

  def _update_constraints(self, user_msg, response):
    """Extract constraints and rejections."""
    try:
      update = self.llm.generate_json(
        system_prompt=(
          "Extract any new constraints or rejected "
          "items from this exchange. Respond as JSON: "
          '{"constraints": {}, "rejected": []}'
        ),
        user_message=(
          f"User: {user_msg}\n"
          f"Assistant: {response}"
        )
      )
      self.active_constraints.update(
        update.get("constraints", {})
      )
      self.rejected_items.update(
        update.get("rejected", [])
      )
    except (json.JSONDecodeError, Exception):
      pass  # Non-critical — constraints are best-effort

#A Keep 6 most recent turns (3 exchanges) in full
#B Compress when conversation exceeds threshold
#C Summarization preserves preference signals
#D Dict for development; use Redis/DB in production
#E Persist updated profile
#F Track rejections to avoid re-recommending
#G Replace last user message with candidate-augmented version
