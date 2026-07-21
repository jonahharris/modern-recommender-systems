# Figure — Listing 10.4: RAG recommendation chain
# Source: chapters/ch10.md lines 223-233
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Generate retrieval-optimized rewrites.
#   #B Format each candidate with its metadata so the LLM can reason about it
def rewrite_query(llm, user_query, n_rewrites=3):
    response = llm.generate(
        system_prompt="You help a movie search engine. "
            "Rewrite the user's query into concrete movie "
            "descriptions that would match well in a "
            "semantic search. Generate exactly "
            f"{n_rewrites} rewrites, one per line. "
            "Do not include movie titles.",
        user_message=user_query
    )
    return response.strip().split("\n")
