"""Model-size ladder (head dim = 64 everywhere) and torch-free size/FLOPs formulas.

Non-embedding params: ~0.4M .. ~59M.  Everything here is pure Python so that grid.py and
fit.py run without torch.
"""
SIZES = {
    # t0/t1 exist because with a 50k vocab the LM head makes tiny models expensive; at small budgets the
    # compute-optimal model can fall BELOW s0, which leaves the IsoFLOP parabola without an interior minimum.
    "t0": dict(n_layer=2, d_model=128, n_head=2),
    "t1": dict(n_layer=3, d_model=192, n_head=3),
    "s0": dict(n_layer=4, d_model=256, n_head=4),
    "s1": dict(n_layer=6, d_model=320, n_head=5),
    "s2": dict(n_layer=8, d_model=384, n_head=6),
    "s3": dict(n_layer=10, d_model=512, n_head=8),
    "s4": dict(n_layer=12, d_model=640, n_head=10),
}
SEQ_LEN = 512
BATCH_TOKENS = 32768  # tokens per optimizer step (all runs)
VOCAB_SIZE = 50304  # GPT-2 BPE (50257) padded to a multiple of 64


def flops_per_token(n_layer, d_model, seq_len=SEQ_LEN, vocab=VOCAB_SIZE):
    """Training FLOPs/token (fwd+bwd = 3x fwd): 6*(12*L*d^2 + d*V) + 6*L*T*d."""
    return 6.0 * (12 * n_layer * d_model**2 + d_model * vocab) + 6.0 * n_layer * seq_len * d_model


def nonembed_params(n_layer, d_model):
    """Blocks (12*L*d^2) + LayerNorms; matches model.count_params()['nonembed'] to <0.1%."""
    return 12 * n_layer * d_model**2 + n_layer * 2 * 2 * d_model + 2 * d_model
