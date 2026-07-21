# Figure — Listing 6.4: Evaluation of the three SASRec ablations
# Source: chapters/ch06.md lines 288-304
# Chapter: 6
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Permute the order of the sequences.
#   #B Remove all but the last item of the input sequence.
#   #C The user state vector is the hidden state at the rightmost position.
#   #D Scoring against every item is just a matrix multiplication against the embedding table.
def evaluate_ablation(model, eval_sequences, eval_targets, mode='full'):
    model.eval()
    sequences = eval_sequences.clone()
    if mode == 'shuffled':
        for i in range(sequences.shape[0]):
            valid = sequences[i] != 0
            shuffled = sequences[i, valid][torch.randperm(valid.sum())]
            sequences[i, valid] = shuffled
    elif mode == 'last_only':
        last_item = sequences.gather(1, (sequences != 0).sum(1, keepdim=True) - 1)
        sequences = torch.zeros_like(sequences)
        sequences[:, -1] = last_item.squeeze(1)
    with torch.no_grad():
        hidden = model(sequences)
        user_state = hidden[:, -1, :]
        scores = user_state @ model.item_emb.weight.T
    return compute_ndcg_at_k(scores, eval_targets, k=10)
