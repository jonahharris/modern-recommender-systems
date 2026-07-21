# Figure - Listing 8.11: Generating semantic IDs
# Source: chapters/ch08.md lines 755-773
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def inference(self, df, data):
  self.rqvae.eval()  #A
  if not isinstance(data, torch.Tensor):  #B
    data = torch.tensor(data,
      dtype=torch.float32).to(self.device)
  with torch.no_grad():
    _, _, codes = self.rqvae(data)  #C
  df = df.copy()  #D
  df['semantic_id'] = [
    tuple(c.cpu().numpy().tolist())
    for c in codes]  #E
  df['_semantic_id_str'] = df['semantic_id'].astype(str)
  df = df.sort_values(
    by=['_semantic_id_str', 'title'])
  df['leaf_id'] = df.groupby(
    '_semantic_id_str').cumcount()  #F
  df = df.drop(columns=['_semantic_id_str'])
  df['final_id'] = df.apply(
    lambda x: x['semantic_id']
    + (x['leaf_id'],), axis=1)  #G
  return df

# Callout annotations (from the book):
#   #A Switch to evaluation mode
#   #B Handle both tensor and array input
#   #C Generate semantic IDs for all items
#   #D Don't modify the original dataframe
#   #E Add semantic IDs as tuples
#   #F Count items within each semantic ID group
#   #G Append leaf node for uniqueness
