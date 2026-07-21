# Figure — Listing 8.6: Prepare text for embedding
# Source: chapters/ch08.md lines 560-568
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Iterate through the dataframe
#   #B Combine title, genres, and description
#   #C Array of strings ready for BERT
def prepare_data(df):
  texts = []
  for _, row in df.iterrows():
    t = str(row['title'])
    genres = str(row['genres'])
    blob = f"{t}. {genres}. " \
      f"{row.get('description', '')}"
    texts.append(blob)
  return texts
