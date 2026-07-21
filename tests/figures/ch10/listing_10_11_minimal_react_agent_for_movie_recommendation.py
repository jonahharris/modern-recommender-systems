# Figure - Listing 10.11: Minimal ReAct agent for movie recommendation
# Source: chapters/ch10.md lines 455-568
# Chapter: 10
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
import json
import re

class MovieRecommenderAgent:
    def __init__(self, llm, tools, max_steps=5):
        self.llm = llm
        self.tools = {t["name"]: t for t in tools}
        self.max_steps = max_steps
        self.trace = []  #A

    def run(self, user_query, user_id=None):
        """Execute the agent loop."""
        system = self._build_system_prompt()
        messages = [{"role": "user", "content": user_query}]
        self.trace = []

        for step in range(self.max_steps):
            response = self.llm.generate(
                system_prompt=system, messages=messages
            )

            if self._is_final_answer(response):  #B
                answer = response.split("Final Answer:")[-1].strip()
                self.trace.append({"type": "final_answer",
                                   "content": answer})
                return answer

            thought, tool_name, args = self._parse_action(response)
            self.trace.append({"type": "thought",
                               "content": thought})  #C

            if tool_name not in self.tools:
                messages.append({"role": "assistant",
                                 "content": response})
                messages.append({"role": "user",
                    "content": f"Error: unknown tool '{tool_name}'. "
                               f"Available: {list(self.tools.keys())}"})
                self.trace.append({"type": "error",
                    "content": f"Unknown tool: {tool_name}"})
                continue

            try:
                result = self.tools[tool_name]["function"](**args)
            except Exception as e:
                result = {"error": str(e)}

            self.trace.append({"type": "action",
                               "tool": tool_name,
                               "args": args})
            self.trace.append({"type": "observation",
                               "content": result})

            messages.append(
                {"role": "assistant", "content": response}
            )
            messages.append(  #D
                {"role": "user",
                 "content": f"Observation: "
                            f"{json.dumps(result, default=str)}"}
            )

        return ("I wasn't able to find a good recommendation. "
                "Could you rephrase your request?")  #E

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
            r"Thought:\s*(.+?)(?=Action:|$)",  #F
            response, re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(
            r"Action:\s*(\w+)\((.+)\)",  #G
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

# Callout annotations (from the book):
#   #A Trace accumulates every step — thoughts, actions, observations, errors
#   #B If the LLM has enough information, it produces a final answer and we extract it
#   #C Every thought, action, and observation is logged to the trace
#   #D Append both the action and its result to the conversation so the LLM can reason about them
#   #E Safety valve — if the agent cannot converge within max\_steps, admit it
#   #F Extract the reasoning between "Thought:" and "Action:"
#   #G Parse the tool call — expects tool\_name({"key": "value"}) format
