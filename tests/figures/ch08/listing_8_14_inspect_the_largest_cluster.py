# Figure — Listing 8.14: Inspect the largest cluster
# Source: chapters/ch08.md lines 853-856
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
largest_id = df.value_counts(
  'semantic_id').head(1).index.values[0]
df[df['semantic_id'] == largest_id][
  ['title', 'genres', 'semantic_id']].head(5)
