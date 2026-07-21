# Figure — Listing 8.11: Generating semantic IDs
# Source: chapters/ch08.md lines 755-773
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Switch to evaluation mode
#   #B Handle both tensor and array input
#   #C Generate semantic IDs for all items
#   #D Don't modify the original dataframe
#   #E Add semantic IDs as tuples
#   #F Count items within each semantic ID group
#   #G Append leaf node for uniqueness
def inference(self, df, data):
  self.rqvae.eval()
  if not isinstance(data, torch.Tensor):
    data = torch.tensor(data,
      dtype=torch.float32).to(self.device)
  with torch.no_grad():
    _, _, codes = self.rqvae(data)
  df = df.copy()
  df['semantic_id'] = [
    tuple(c.cpu().numpy().tolist())
    for c in codes]
  df = df.sort_values(
    by=['semantic_id', 'title'])
  df['leaf_id'] = df.groupby(
    'semantic_id').cumcount()
  df['final_id'] = df.apply(
    lambda x: x['semantic_id']
    + (x['leaf_id'],), axis=1)
  return df
