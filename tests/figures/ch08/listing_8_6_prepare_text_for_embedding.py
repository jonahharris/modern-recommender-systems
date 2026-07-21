# Figure - Listing 8.6: Prepare text for embedding
# Source: chapters/ch08.md lines 560-568
# Chapter: 8
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def prepare_data(df):
  texts = []
  for _, row in df.iterrows():  #A
    t = str(row['title'])
    genres = str(row['genres'])
    blob = f"{t}. {genres}. " \
      f"{row.get('description', '')}"  #B
    texts.append(blob)
  return texts  #C

# Callout annotations (from the book):
#   #A Iterate through the dataframe
#   #B Combine title, genres, and description
#   #C Array of strings ready for BERT
