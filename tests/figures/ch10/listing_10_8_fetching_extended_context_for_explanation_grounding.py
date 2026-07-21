# Figure — Listing 10.8: Fetching extended context for explanation grounding
# Source: chapters/ch10.md lines 362-370
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Pull from a document store, knowledge graph, or API — anything richer than the vector index payload
def enrich_candidates(candidates, document_store):
    enriched = []
    for item in candidates:
        doc = document_store.get(item["movie_id"])
        item["full_description"] = doc.get("description", "")
        item["reviews_summary"] = doc.get("reviews_summary", "")
        item["awards"] = doc.get("awards", "")
        enriched.append(item)
    return enriched
