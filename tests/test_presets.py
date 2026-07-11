from app.model import MULTITASK_VOCAB, ModelConfig, TinyTransformer
from app.training import PRESETS, TRAINING_MODES


def test_presets_are_valid_and_parameter_claims_match():
    ids = set()
    for preset in PRESETS:
        assert preset["id"] not in ids
        ids.add(preset["id"])
        assert preset["operation"] in TRAINING_MODES
        assert preset["d_model"] % preset["n_heads"] == 0
        model = TinyTransformer(ModelConfig(
            seq_len=preset.get("seq_len", 9),
            vocab_size=len(MULTITASK_VOCAB) if preset["operation"] in ("multi_task", "translate") else 12,
            d_model=preset["d_model"], n_heads=preset["n_heads"],
            n_layers=preset["n_layers"], d_ff=preset["d_ff"]
        ))
        expected = sum(p.numel() for p in model.parameters())
        assert f"{expected:,}" in preset["description"]
