"""
Minimal ReAct agent for movie recommendation.

Implements the Thought -> Action -> Observation loop in pure Python,
without external agent frameworks, to keep the mechanism transparent.
"""

import json
import re


class Tool:
  """A callable tool the agent can invoke."""

  def __init__(self, name, description, function,
               parameters=None):
    self.name = name
    self.description = description
    self.function = function
    self.parameters = parameters or {}

  def __call__(self, **kwargs):
    return self.function(**kwargs)


class MovieRecommenderAgent:

  def __init__(self, llm, tools, max_steps=5):
    self.llm = llm
    # Accept both Tool objects and dicts
    if tools and isinstance(tools[0], Tool):
      self.tools = {t.name: {"name": t.name,
        "description": t.description,
        "function": t.function,
        "parameters": t.parameters} for t in tools}
    else:
      self.tools = {t["name"]: t for t in tools}
    self.max_steps = max_steps
    self.trace = []                                    #A

  def run(self, user_query, user_id=None, debug=False):
    """Execute the agent loop."""
    system = self._build_system_prompt()
    self.trace = []

    if user_id is not None:
      user_query = f"[User ID: {user_id}] {user_query}"
    messages = [{"role": "user", "content": user_query}]

    for step in range(self.max_steps):                 #B
      if debug:
        print(f"\n--- Step {step + 1}: "
              f"Calling LLM... ---", flush=True)

      try:
        response = self.llm.generate(
          system_prompt=system, messages=messages
        )
      except Exception as e:
        if debug:
          print(f"LLM error: {e}", flush=True)
        return f"Error calling LLM: {e}"

      if not response:
        if debug:
          print("Empty response from LLM", flush=True)
        continue

      if debug:
        preview = response[:150].replace("\n", " ")
        print(f"Response ({len(response)} chars): "
              f"{preview}...", flush=True)

      # Check for final answer
      if self._is_final_answer(response):
        answer = response.split(
          "Final Answer:", 1
        )[1].strip()
        self.trace.append({
          "step": step + 1,
          "type": "final_answer",
          "content": answer
        })
        if debug:
          print(f"=> Final answer received",
                flush=True)
        return answer

      # Try to parse a tool call
      thought, tool_name, args = self._parse_action(
        response
      )

      if tool_name is None or tool_name not in self.tools:
        # If we already have observations and the LLM
        # responded without "Final Answer:" prefix,
        # treat the response as a final answer.        #C
        has_observations = any(
          m["role"] == "user"
          and m["content"].startswith("Observation:")
          for m in messages
        )
        if has_observations and len(response) > 50:
          self.trace.append({
            "step": step + 1,
            "type": "final_answer",
            "content": response
          })
          if debug:
            print(f"=> Implicit final answer "
                  f"(no prefix)", flush=True)
          return response

        # First step or very short response
        self.trace.append({
          "step": step + 1,
          "type": "error",
          "thought": thought,
          "raw_response": response[:200]
        })
        if debug:
          print(f"=> No valid action found, "
                f"asking LLM to retry", flush=True)
        messages.append(
          {"role": "assistant", "content": response}
        )
        messages.append(
          {"role": "user",
           "content": "Please use a tool first. "
           f"Available: {list(self.tools.keys())}. "
           "Format: Action: tool_name("
           "{\"arg\": \"val\"})"}
        )
        continue

      # Execute the tool
      if debug:
        print(f"=> Calling {tool_name}({args})",
              flush=True)

      try:
        observation = self.tools[tool_name][
          "function"
        ](**args)
      except Exception as e:
        observation = {"error": str(e)}

      self.trace.append({                              #D
        "step": step + 1,
        "type": "action",
        "thought": thought,
        "tool": tool_name,
        "args": args,
        "observation": observation
      })

      if debug:
        if isinstance(observation, list):
          print(f"=> Got {len(observation)} results",
                flush=True)
        else:
          obs_str = str(observation)[:100]
          print(f"=> Observation: {obs_str}",
                flush=True)

      messages.append(                                 #E
        {"role": "assistant", "content": response}
      )
      # Truncate observation to save context tokens
      obs_str = json.dumps(observation, default=str)
      if len(obs_str) > 2000:                          #F
        obs_str = obs_str[:2000] + "... (truncated)"
      messages.append(
        {"role": "user",
         "content": f"Observation: {obs_str}"}
      )

    return (
      "I wasn't able to find a good recommendation. "
      "Could you rephrase your request?"
    )

  def _build_system_prompt(self):
    tool_descriptions = "\n".join(
      f"- {t['name']}: {t['description']}"
      for t in self.tools.values()
    )
    return f"""You are a movie recommendation agent.
Think step by step to answer the user's request.

Available tools:
{tool_descriptions}

At each step, respond with either:
1. A thought and action in this exact format:
   Thought: <your reasoning>
   Action: <tool_name>(<arguments as JSON>)
2. A final answer:
   Final Answer: <your recommendation with explanation>

Always ground recommendations in tool results.
Never recommend a movie you haven't retrieved."""

  def _is_final_answer(self, response):
    return "Final Answer:" in response

  def _parse_action(self, response):
    """Extract thought, tool name, and arguments."""
    thought = ""
    thought_match = re.search(
      r"Thought:\s*(.+?)(?=Action:|$)",                #F
      response, re.DOTALL
    )
    if thought_match:
      thought = thought_match.group(1).strip()

    action_match = re.search(
      r"Action:\s*(\w+)\((.+)\)",                      #G
      response, re.DOTALL
    )
    if not action_match:
      return thought, None, {}

    tool_name = action_match.group(1)
    try:
      args = json.loads(action_match.group(2))
    except json.JSONDecodeError:
      args = {}

    return thought, tool_name, args

  def get_trace(self):
    """Return the reasoning trace for debugging."""
    return self.trace

#A Trace records one entry per step
#B Loop up to max_steps to prevent runaway chains
#C Accept response as final answer if tools already returned results
#D Each successful step records thought, action, and observation together
#E Append to messages so the LLM sees the conversation history
#F Extract reasoning between "Thought:" and "Action:"
#G Parse tool call -- expects tool_name({"key": "value"}) format
