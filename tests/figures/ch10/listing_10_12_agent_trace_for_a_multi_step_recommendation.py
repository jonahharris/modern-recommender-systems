# Figure — Listing 10.12: Agent trace for a multi-step recommendation
# Source: chapters/ch10.md lines 594-603
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
agent = MovieRecommenderAgent(
    llm=llm,
    tools=tools
)

response = agent.run(
    "Find me a sci-fi movie from the 1990s similar to Dune"
)

print(response)
