# Figure - Listing 10.12: Agent trace for a multi-step recommendation
# Source: chapters/ch10.md lines 594-603
# Chapter: 10
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
agent = MovieRecommenderAgent(
    llm=llm,
    tools=tools
)

response = agent.run(
    "Find me a sci-fi movie from the 1990s similar to Dune"
)

print(response)
