# Figure - Listing 6.1: Sequential base class
# Source: chapters/ch06.md lines 132-180
# Chapter: 6
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class _SequentialBase(nn.Module):
  def __init__(
    self,
    num_items: int,
    max_len: int = 200,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
    extra_tokens: int = 0,  #A
  ) -> None:

    super().__init__()
    self.num_items = num_items
    self.max_len = max_len
    vocab_size = num_items + 1 + extra_tokens
    self.item_emb = nn.Embedding(vocab_size, hidden_dim,
                                 padding_idx=0)  #B
    self.pos_emb  = nn.Embedding(max_len, hidden_dim)  #C
    self.input_dropout = nn.Dropout(dropout)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=hidden_dim,
      nhead=num_heads,
      dim_feedforward=hidden_dim,
      dropout=dropout,
      batch_first=True,
      norm_first=True,  #D
      )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    self.final_norm  = nn.LayerNorm(hidden_dim)

  def _encode(
    self,
    sequences: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

    batch_size, seq_len = sequences.shape
    positions = torch.arange(seq_len, device=sequences.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    x = self.item_emb(sequences) + self.pos_emb(positions)
    x = self.input_dropout(x)
    padding_mask = (sequences == 0)  #E

    for layer in self.transformer.layers:  #F
      x = layer(x, src_mask=attn_mask,
                src_key_padding_mask=padding_mask)
      x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  #G
    return self.final_norm(x)

# Callout annotations (from the book):
#   #A Room for special tokens beyond padding.
#   #B Item embeddings, with index 0 reserved for the padding token; real items start at 1\.
#   #C Learned positional embeddings, one per absolute position in the window.
#   #D Pre-norm layout: LayerNorm runs before attention, which is more stable to train.
#   #E True wherever the sequence is padding; attention will ignore those positions as keys.
#   #F The layers are applied one at a time.
#   #G After each layer, representations at padding positions are overwritten with zeros.
