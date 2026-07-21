# Figure - Listing 8.12: Find all Star Trek movies
# Source: chapters/ch08.md lines 817-823
# Chapter: 8
# Category: needs-real-data  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
columns = ['title', 'genres',
  'description', 'semantic_id']
star_treks = df[
  df['title'].str.startswith('Star Trek')
][columns]
star_treks.sort_values(
  by=['semantic_id']).head(5)
