# Figure - Listing 8.8: Running the pipeline
# Source: chapters/ch08.md lines 634-642
# Chapter: 8
# Category: needs-real-data  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
texts = prepare_data(df)  #A
embeddings = pipeline.text_encoder.encode(
  texts, show_progress_bar=True)  #B
data_tensor = \
  pipeline.initialize_data_with_embeddings(
    embeddings)  #C
pipeline.train(data_tensor, epochs=500)  #D
df_enriched = pipeline.inference(
  df, data_tensor)  #E

# Callout annotations (from the book):
#   #A Prepare text from titles, genres, descriptions
#   #B Encode all items with BERT
#   #C Initialize codebooks with K-means
#   #D Train the RQ-VAE
#   #E Generate semantic IDs
