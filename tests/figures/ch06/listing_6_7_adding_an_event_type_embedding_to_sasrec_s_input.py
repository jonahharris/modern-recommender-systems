# Figure - Listing 6.7: Adding an event-type embedding to SASRec's input
# Source: chapters/ch06.md lines 493-521
# Chapter: 6
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class SASRecWithEvents(SASRec):
  def __init__(self,
               num_items,
               num_event_types,
               max_len=200,
               hidden_dim=64,
               num_layers=2,
               num_heads=2, dropout=0.2):
    super().__init__(num_items,
                     max_len,
                     hidden_dim,
                     num_layers, num_heads, dropout):
    self.event_emb = nn.Embedding(num_event_types, hidden_dim)  #A

  def forward(self, sequences, event_types):
    batch_size, seq_len = sequences.shape
    positions = torch.arange(seq_len, device=sequences.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    x = (self.item_emb(sequences)
         + self.pos_emb(positions)
         + self.event_emb(event_types))  #B
    x = self.input_dropout(x)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=sequences.device, dtype=torch.bool),
        diagonal=1,
    )
    padding_mask = (sequences == 0)
    x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
    return self.final_norm(x)

# Callout annotations (from the book):
#   #A One embedding vector per event type, learned jointly with everything else.
#   #B The only change to the forward pass: a third term added to the input, exactly as positional embeddings were added in Listing 6.1.
