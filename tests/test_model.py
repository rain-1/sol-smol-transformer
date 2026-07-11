import torch

from app.model import LETTER_IDS, MULTITASK_VOCAB, OPERATION_IDS, PAD_ID, ModelConfig, TinyTransformer, decode, encode
from app.training import make_batch, make_multitask_batch, transform


def test_operations():
    x = torch.tensor([[1, 4, 2, 3]])
    assert transform(x, "copy").tolist() == [[1, 4, 2, 3]]
    assert transform(x, "reverse").tolist() == [[3, 2, 4, 1]]
    assert transform(x, "sort").tolist() == [[1, 2, 3, 4]]
    assert transform(x, "rotate_left").tolist() == [[4, 2, 3, 1]]


def test_model_shapes_and_internals():
    cfg = ModelConfig(seq_len=6, d_model=16, n_heads=2, n_layers=2, d_ff=24)
    model = TinyTransformer(cfg)
    logits, layers = model(torch.randint(0, 10, (3, 6)), capture=True)
    assert logits.shape == (3, 6, 12)
    assert len(layers) == 2
    assert layers[0]["attention"].shape == (3, 2, 6, 6)
    assert layers[0]["mlp"].shape == (3, 6, 24)


def test_codec_validation():
    assert decode(encode("0123", 5)) == "0123="
    assert decode(encode("abc", 4)) == "abc="
    assert decode(encode("012=", 6, pad=True)) == "012="
    try:
        encode("abd", 4)
        assert False
    except ValueError:
        pass


def test_variable_length_batch_preserves_equals_and_padding():
    x, y = make_batch(64, 9, "reverse", torch.device("cpu"))
    assert x.shape == y.shape == (64, 9)
    for source, target in zip(x, y):
        end = (source == 10).nonzero()[0].item()
        assert target[end].item() == 10
        assert target[:end].tolist() == source[:end].flip(0).tolist()
        assert target[end + 1:].eq(PAD_ID).all()


def test_multitask_batches_have_operation_prefixes():
    x, y = make_multitask_batch(128, 10, torch.device("cpu"))
    assert set(x[:, 0].tolist()).issubset(set(OPERATION_IDS.values()))
    assert y[:, 0].eq(PAD_ID).all()
    assert x[:, 1:].eq(10).any(dim=1).all()


def test_translate_maps_both_alphabets():
    letters = torch.tensor([[LETTER_IDS["a"], LETTER_IDS["b"], LETTER_IDS["c"]]])
    digits = torch.tensor([[1, 2, 3]])
    assert transform(digits, "translate").tolist() == letters.tolist()
    assert transform(letters, "translate").tolist() == digits.tolist()
    x, y = make_batch(128, 9, "translate", torch.device("cpu"))
    valid = y.ne(PAD_ID)
    assert x[valid].max() < len(MULTITASK_VOCAB)
