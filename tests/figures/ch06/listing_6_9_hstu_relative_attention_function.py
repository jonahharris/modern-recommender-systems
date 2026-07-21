# Figure - Listing 6.9: HSTU relative attention function
# Source: chapters/ch06.md lines 585-607
# Chapter: 6
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class RelativeAttention(nn.Module):
    '''
    Simplified illustration of HSTU's relative attention.
    Rather than adding position embeddings to token representations,
    the distance between positions is encoded directly in each
    attention weight — so the model can use sequence length as a
    scaling dimension rather than a fixed architectural constraint.
    '''
    def __init__(self, hidden_dim, max_relative_dist=200):
        super().__init__()
        self.rel_emb = nn.Embedding(max_relative_dist * 2 + 1, hidden_dim)  # A
        self.scale   = hidden_dim ** 0.5

    def forward(self, queries, keys, positions):
        # queries, keys : (batch, seq_len, hidden_dim)
        # positions      : (batch, seq_len) — absolute position indices
        seq_len = queries.shape[1]
        content_score = torch.bmm(queries, keys.transpose(1, 2)) / self.scale

        # Relative distance between every pair of positions.
        rel_dist  = positions.unsqueeze(2) - positions.unsqueeze(1)          # B
        rel_score = torch.bmm(queries, self.rel_emb(rel_dist + 200).transpose(1, 2)) / self.scale
        return F.softmax(content_score + rel_score, dim=-1)

# Callout annotations (from the book):
#   #A  One embedding per relative distance bucket, positive and negative.
#   #B  rel\_dist\[i, j\] \= position\_i \- position\_j; the model learns what  near-vs-far means from data rather than from a hand-coded positional scheme.
