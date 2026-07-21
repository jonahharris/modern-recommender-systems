# Figure - Listing 6.10: OnTrans Tokenizer
# Source: chapters/ch06.md lines 626-648
# Chapter: 6
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class OneTransTokenizer(nn.Module):
    '''
    Converts heterogeneous inputs — sequential item history and
    non-sequential user/context features — into a unified token stream.
    '''
    def __init__(self, num_items, num_features, hidden_dim):
        super().__init__()
        self.item_emb    = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.feature_emb = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_dim) for vocab_size in num_features
        ])
        self.feature_proj = nn.Linear(len(num_features) * hidden_dim, hidden_dim)  # A

    def forward(self, item_seq, feature_ids):
        # item_seq    : (batch, seq_len) — sequential behavior tokens
        # feature_ids : (batch, num_features) — non-sequential feature indices
        seq_tokens     = self.item_emb(item_seq)              # (batch, seq_len, hidden_dim)
        feat_embeddings = torch.cat(
            [emb(feature_ids[:, i]) for i, emb in enumerate(self.feature_emb)],
            dim=-1,
        )                                                      # (batch, num_features * hidden_dim)
        feat_token = self.feature_proj(feat_embeddings).unsqueeze(1)  # (batch, 1, hidden_dim)  # B
        return torch.cat([seq_tokens, feat_token], dim=1)     # (batch, seq_len + 1, hidden_dim)

# Callout annotations (from the book):
#   #A Non-sequential features are projected into the same hidden dimension as item tokens — a single linear layer that compresses all feature embeddings into one vector.
#   #B The compressed feature vector becomes one additional token appended to the behavior sequence, so the same transformer stack processes both without architectural branching.
