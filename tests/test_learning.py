import torch
from torch import nn

from app.model import PAD_ID, ModelConfig, TinyTransformer
from app.training import make_batch, transform


def test_reverse_can_overfit_tiny_batch():
    torch.manual_seed(1)
    model = TinyTransformer(ModelConfig(seq_len=4, d_model=16, n_heads=2, n_layers=1, d_ff=32))
    x = torch.randint(0, 10, (16, 4)); y = transform(x, "reverse")
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(100):
        logits, _ = model(x); loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    assert (model(x)[0].argmax(-1) == y).float().mean() > 0.98


def test_attention_masks_padding():
    model = TinyTransformer(ModelConfig(seq_len=6, d_model=8, n_heads=1, n_layers=1, d_ff=8))
    x, _ = make_batch(8, 6, "copy", torch.device("cpu"))
    _, layers = model(x, capture=True)
    padded_keys = x.eq(PAD_ID)[:, None, None, :].expand_as(layers[0]["attention"])
    assert layers[0]["attention"].masked_select(padded_keys).abs().max() < 1e-6
