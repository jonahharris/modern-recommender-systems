"""
Guardrails for agentic recommendation:
- GroundingValidator: ensures recommendations exist in catalog
- LoopDetector: catches repeated tool calls  
- AgentEvaluator: evaluation harness
"""

import json
import re
import time
from typing import List, Dict, Set, Optional


class GroundingValidator:
  """Validate that agent recommendations exist in the catalog."""

  def __init__(self, catalog_df, title_column="title"):
    self.titles = set(
      catalog_df[title_column].str.lower().str.strip()
    )

  def validate(self, response, candidates=None):
    """Check that recommendations are grounded.
    
    Args:
      response: Agent's natural language response
      candidates: Optional list of retrieved candidates
      
    Returns:
      dict with grounded, hallucinated, and ungrounded items
    """
    mentioned = self._extract_titles(response)
    candidate_titles = set()
    if candidates:
      candidate_titles = {
        c["title"].lower().strip() for c in candidates
      }

    grounded = []
    hallucinated = []
    ungrounded = []

    for title in mentioned:
      title_lower = title.lower().strip()
      if title_lower not in self.titles:               #A
        hallucinated.append(title)
      elif candidates and title_lower not in candidate_titles:
        ungrounded.append(title)                       #B
      else:
        grounded.append(title)

    return {
      "grounded": grounded,
      "hallucinated": hallucinated,
      "ungrounded": ungrounded,
      "grounding_rate": (
        len(grounded) / len(mentioned) 
        if mentioned else 1.0
      )
    }

  def _extract_titles(self, response):
    """Extract movie titles from agent response.
    
    Looks for common patterns:
    - **Title (Year)** — bold markdown
    - "Title (Year)" — quoted
    - N. Title (Year) — numbered list
    """
    patterns = [
      r'\*\*(.+?)\s*\(\d{4}\)\*\*',
      r'"(.+?)\s*\(\d{4}\)"',
      r'\d+\.\s*\*?\*?(.+?)\s*\(\d{4}\)',
      r'(?:^|\n)\s*[-•]\s*(.+?)\s*\(\d{4}\)',
    ]
    titles = set()
    for pattern in patterns:
      for match in re.finditer(pattern, response):
        title = match.group(1).strip().strip("*")
        if len(title) > 1 and len(title) < 100:
          titles.add(title)
    return list(titles)


class LoopDetector:
  """Detect when an agent is repeating tool calls."""

  def __init__(self, window=6):
    self.window = window

  def is_looping(self, trace):
    """Check recent trace for repeated actions."""
    if len(trace) < 3:
      return False

    recent = trace[-self.window:]
    actions = []
    for step in recent:
      resp = step.get("response", "")
      match = re.search(r'Action:\s*(\w+\(.*?\))', 
                        resp, re.DOTALL)
      if match:
        actions.append(match.group(1))

    if len(actions) < 2:
      return False

    unique = set(actions)
    return len(unique) < len(actions)                  #C


class AgentEvaluator:
  """Evaluation harness for agentic recommendation."""

  def __init__(self, catalog_titles=None):
    self.catalog_titles = catalog_titles or set()
    self.validator = None
    if catalog_titles:
      # Build a minimal validator
      import pandas as pd
      df = pd.DataFrame({"title": list(catalog_titles)})
      self.validator = GroundingValidator(df)

  def evaluate_single(self, agent, test_case):
    """Evaluate a single test case.
    
    Args:
      agent: MovieRecommenderAgent or 
             ConversationalRecommender
      test_case: dict with query, relevant_items, 
                 and optional user_id
                 
    Returns:
      dict with metrics
    """
    start = time.time()

    if hasattr(agent, "chat"):
      response = agent.chat(
        test_case.get("user_id"), test_case["query"]
      )
    else:
      response = agent.run(
        test_case["query"], test_case.get("user_id")
      )

    elapsed = time.time() - start

    # Extract recommended titles
    recommended = self._extract_titles(response)
    relevant = set(
      t.lower() for t in test_case.get("relevant_items", [])
    )

    # Compute metrics
    rec_lower = [t.lower() for t in recommended]
    hits = len(set(rec_lower) & relevant)

    precision = (
      hits / len(recommended) if recommended else 0
    )
    recall = hits / len(relevant) if relevant else 0

    # Grounding check
    grounding = {"grounding_rate": 1.0}
    if self.validator:
      grounding = self.validator.validate(response)

    # Step count
    num_steps = 0
    if hasattr(agent, "get_trace"):
      num_steps = len(agent.get_trace())

    return {
      "query": test_case["query"],
      "recommended": recommended,
      "precision": precision,
      "recall": recall,
      "grounding_rate": grounding["grounding_rate"],
      "hallucinated": grounding.get("hallucinated", []),
      "latency_seconds": elapsed,
      "num_steps": num_steps,
      "response": response
    }

  def evaluate_batch(self, agent, test_cases):
    """Run evaluation on a batch of test cases."""
    results = []
    for case in test_cases:
      result = self.evaluate_single(agent, case)
      results.append(result)

    # Aggregate
    n = len(results)
    avg_precision = sum(
      r["precision"] for r in results
    ) / n
    avg_recall = sum(
      r["recall"] for r in results
    ) / n
    avg_grounding = sum(
      r["grounding_rate"] for r in results
    ) / n
    avg_latency = sum(
      r["latency_seconds"] for r in results
    ) / n
    avg_steps = sum(
      r["num_steps"] for r in results
    ) / n

    return {
      "individual": results,
      "aggregate": {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_grounding_rate": avg_grounding,
        "avg_latency_seconds": avg_latency,
        "avg_steps": avg_steps,
        "n_test_cases": n
      }
    }

  def evaluate_multiturn(self, recommender, scenario):
    """Evaluate a multi-turn conversation scenario.
    
    Args:
      recommender: ConversationalRecommender
      scenario: dict with user_id and turns, where each
                turn has user_message and assertions
                
    Returns:
      list of per-turn results
    """
    recommender.reset()
    results = []

    for i, turn in enumerate(scenario["turns"]):
      response = recommender.chat(
        scenario["user_id"], turn["user_message"]
      )

      turn_result = {
        "turn": i + 1,
        "user_message": turn["user_message"],
        "response": response,
        "checks": []
      }

      for check in turn.get("assertions", []):
        passed = check["fn"](response)
        turn_result["checks"].append({
          "name": check["name"],
          "passed": passed
        })

      results.append(turn_result)

    return results

  def _extract_titles(self, response):
    """Extract movie titles from response text."""
    patterns = [
      r'\*\*(.+?)\s*\(\d{4}\)\*\*',
      r'"(.+?)\s*\(\d{4}\)"',
      r'\d+\.\s*\*?\*?(.+?)\s*\(\d{4}\)',
      r'(?:^|\n)\s*[-•]\s*(.+?)\s*\(\d{4}\)',
    ]
    titles = []
    seen = set()
    for pattern in patterns:
      for match in re.finditer(pattern, response):
        title = match.group(1).strip().strip("*")
        if title.lower() not in seen and len(title) > 1:
          titles.append(title)
          seen.add(title.lower())
    return titles


def llm_as_judge(llm, query, recommendation, 
                  explanation, user_profile=""):
  """Score an explanation using LLM-as-judge.
  
  Returns:
    dict with specificity, accuracy, helpfulness scores (1-5)
  """
  score_response = llm.generate_json(
    system_prompt="""You evaluate recommendation 
explanations. Score each explanation 1-5 on:
- Specificity: Does it reference concrete attributes?
- Accuracy: Do the stated reasons hold up?
- Helpfulness: Would a user find this useful?

Respond as JSON only:
{"specificity": N, "accuracy": N, 
 "helpfulness": N, "reasoning": "..."}""",
    user_message=(
      f"Query: {query}\n"
      f"Recommendation: {recommendation}\n"
      f"Explanation: {explanation}\n"
      f"User profile: {user_profile}"
    )
  )
  return score_response

#A Item doesn't exist in catalog — fabricated
#B Item exists but wasn't in retrieval results — off-script
#C Duplicate actions in recent window indicate looping
