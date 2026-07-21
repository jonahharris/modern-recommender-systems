# Figure - Listing 10.17: Conversation memory with summarization
# Source: chapters/ch10.md lines 718-755
# Chapter: 10
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class ConversationMemory:
  def __init__(self, llm, max_recent_turns=6,
               summary_threshold=10):
    self.llm = llm
    self.turns = []
    self.summary = ""
    self.max_recent = max_recent_turns  #A
    self.threshold = summary_threshold

  def add_turn(self, role, content):
    self.turns.append({"role": role, "content": content})
    if len(self.turns) > self.threshold:  #B
      self._compress()

  def get_context(self):  #C

    context = []
    if self.summary:
      context.append(
        {"role": "system",
         "content": f"Summary of earlier conversation: "
                    f"{self.summary}"}
      )
    context.extend(self.turns[-self.max_recent:])
    return context

  def _compress(self):  #D
    old_turns = self.turns[:-self.max_recent]
    turns_text = "\n".join(
      f"{t['role']}: {t['content']}" for t in old_turns
    )
    self.summary = self.llm.generate(  #E
      system_prompt="Summarize this conversation "
        "history concisely, preserving the user's "
        "stated preferences and rejected items.",
      user_message=turns_text
    )
    self.turns = self.turns[-self.max_recent:]  #F

# Callout annotations (from the book):
#   #A Keep the 6 most recent turns (3 exchanges) in full fidelity
#   #B When the conversation gets long, compress older turns into a summary
#   #C Returns summary \+ recent turns for prompting.
#   #D Summarize older turns to stay within budget.
#   #E The summarization prompt focuses on preserving preference signals
#   #F Replace the turn list with only recent turns; the summary preserves everything else
