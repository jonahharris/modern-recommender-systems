"""
Thin LLM client wrapper with two backends:
- OpenAI-compatible API (works with OpenAI, Anthropic via proxy, local servers)
- Local HuggingFace transformers (for running without API keys)

Usage:
  # API backend (recommended for agent quality)
  llm = LLMClient(backend="api", api_key="sk-...", model="gpt-4o-mini")
  
  # Local backend (no API key needed, slower)
  llm = LLMClient(backend="local", model="microsoft/Phi-3-mini-4k-instruct")
"""

import json
from typing import Optional


class LLMClient:

  def __init__(self, 
               backend="api",
               model=None,
               api_key=None,
               base_url=None,
               temperature=0.7,
               max_tokens=1024):
    self.backend = backend
    self.temperature = temperature
    self.max_tokens = max_tokens

    if backend == "api":
      self._init_api(model, api_key, base_url)
    elif backend == "local":
      self._init_local(model)
    else:
      raise ValueError(f"Unknown backend: {backend}")

  def _init_api(self, model, api_key, base_url):
    """Initialize OpenAI-compatible API client."""
    try:
      from openai import OpenAI                        #A
    except ImportError:
      raise ImportError(
        "Install openai: pip install openai"
      )
    self.model = model or "gpt-4o-mini"
    self.client = OpenAI(
      api_key=api_key,
      base_url=base_url
    )

  def _init_local(self, model):
    """Initialize local HuggingFace model."""
    try:
      import torch
      from transformers import pipeline
    except ImportError:
      raise ImportError(
        "Install transformers: pip install transformers"
      )
    self.model = model or "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading {self.model}...")
    self._pipeline = pipeline(                         #B
      "text-generation",
      model=self.model,
      torch_dtype=torch.float32,
      device_map="auto"
    )
    print(f"Model loaded.")

  def generate(self, 
               system_prompt,
               user_message=None,
               messages=None):
    """Generate a response.
    
    Args:
      system_prompt: System instructions
      user_message: Single user message (simple usage)
      messages: Full message list (multi-turn usage)
      
    Returns:
      str: The generated text
    """
    if messages is None:
      messages = [{"role": "user", "content": user_message}]

    if self.backend == "api":
      return self._generate_api(system_prompt, messages)
    else:
      return self._generate_local(system_prompt, messages)

  def _generate_api(self, system_prompt, messages):
    full_messages = [
      {"role": "system", "content": system_prompt}
    ] + messages

    response = self.client.chat.completions.create(
      model=self.model,
      messages=full_messages,
      temperature=self.temperature,
      max_tokens=self.max_tokens
    )
    return response.choices[0].message.content

  def _generate_local(self, system_prompt, messages):
    full_messages = [
      {"role": "system", "content": system_prompt}
    ] + messages

    output = self._pipeline(
      full_messages,
      max_new_tokens=self.max_tokens,
      temperature=self.temperature,
      do_sample=True
    )
    return output[0]["generated_text"][-1]["content"]

  def generate_json(self, system_prompt, user_message):
    """Generate and parse a JSON response."""
    json_prompt = (
      system_prompt + 
      "\n\nRespond with valid JSON only. "
      "No markdown, no explanation."
    )
    response = self.generate(json_prompt, user_message)
    # Strip markdown fences if present
    clean = response.strip()
    if clean.startswith("```"):
      clean = clean.split("\n", 1)[1]
      clean = clean.rsplit("```", 1)[0]
    return json.loads(clean)

#A OpenAI client works with any OpenAI-compatible API
#B Local pipeline loads model onto available hardware
