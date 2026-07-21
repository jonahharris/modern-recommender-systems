# Figure — Listing 8.8: Running the pipeline
# Source: chapters/ch08.md lines 634-642
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Prepare text from titles, genres, descriptions
#   #B Encode all items with BERT
#   #C Initialize codebooks with K-means
#   #D Train the RQ-VAE
#   #E Generate semantic IDs
texts = prepare_data(df)
embeddings = pipeline.text_encoder.encode(
  texts, show_progress_bar=True)
data_tensor = \
  pipeline.initialize_data_with_embeddings(
    embeddings)
pipeline.train(data_tensor, epochs=500)
df_enriched = pipeline.inference(
  df, data_tensor)
