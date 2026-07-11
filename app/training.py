from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .model import (LETTER_IDS, MULTITASK_VOCAB, OPERATION_IDS, OPERATION_TOKENS, ModelConfig,
                    PAD_ID, TinyTransformer, VOCAB, decode, encode)


OPERATIONS = ("copy", "reverse", "sort", "rotate_left", "translate")
TRAINING_MODES = OPERATIONS + ("multi_task",)
PRESETS = (
    {"id": "copy_micro", "name": "Copy · micro", "description": "756 params · 100% exact",
     "operation": "copy", "d_model": 8, "n_heads": 1, "n_layers": 1, "d_ff": 8},
    {"id": "reverse_smol", "name": "Reverse · smol", "description": "3,964 params · 100% exact",
     "operation": "reverse", "d_model": 16, "n_heads": 2, "n_layers": 2, "d_ff": 16},
    {"id": "sort_smol", "name": "Sort · smol", "description": "2,268 params · 99.08% exact",
     "operation": "sort", "d_model": 16, "n_heads": 1, "n_layers": 1, "d_ff": 16},
    {"id": "rotate_micro", "name": "Rotate left · micro", "description": "756 params · 100% exact",
     "operation": "rotate_left", "d_model": 8, "n_heads": 1, "n_layers": 1, "d_ff": 8},
    {"id": "translate_smol", "name": "Translate · smol", "description": "2,532 params · 100% exact",
     "operation": "translate", "d_model": 16, "n_heads": 1, "n_layers": 1, "d_ff": 16},
    {"id": "explorer", "name": "Explorer · multi-head", "description": "18,220 params · richer internals",
     "operation": "reverse", "d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64},
    {"id": "multitask", "name": "All tasks · shared", "description": "18,772 params · operation-token routed",
     "operation": "multi_task", "seq_len": 10, "steps": 2000,
     "d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64},
)


def transform(x: torch.Tensor, operation: str) -> torch.Tensor:
    if operation == "copy":
        return x.clone()
    if operation == "reverse":
        return x.flip(-1)
    if operation == "sort":
        return x.sort(dim=-1).values
    if operation == "rotate_left":
        return torch.roll(x, shifts=-1, dims=-1)
    if operation == "translate":
        result = x.clone()
        for digit, letter in zip((1, 2, 3), (LETTER_IDS["a"], LETTER_IDS["b"], LETTER_IDS["c"])):
            result[x == digit] = letter
            result[x == letter] = digit
        return result
    raise ValueError(f"Unknown operation: {operation}")


def make_batch(batch_size: int, max_len: int, operation: str, device: torch.device):
    """Generate mixed-length digit sequences ending in = and padded on the right."""
    if max_len < 3:
        raise ValueError("Maximum sequence length must be at least 3 (two digits plus =)")
    lengths = torch.randint(2, max_len, (batch_size,), device=device)
    x = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full_like(x, PAD_ID)
    for length in range(2, max_len):
        rows = lengths.eq(length)
        count = int(rows.sum())
        if not count:
            continue
        if operation == "translate":
            source_is_letters = torch.randint(0, 2, (count, 1), device=device).bool()
            digits = torch.randint(1, 4, (count, length), device=device)
            letters = torch.tensor([LETTER_IDS["a"], LETTER_IDS["b"], LETTER_IDS["c"]], device=device)[digits - 1]
            payload = torch.where(source_is_letters, letters, digits)
        else:
            payload = torch.randint(0, 10, (count, length), device=device)
        x[rows, :length], x[rows, length] = payload, 10
        y[rows, :length], y[rows, length] = transform(payload, operation), 10
    return x, y


def make_multitask_batch(batch_size: int, max_len: int, device: torch.device):
    """Mix tasks per batch, prefixing each example with its operation token."""
    if max_len < 4:
        raise ValueError("Multi-task maximum length needs operation + two digits + =")
    task_ids = torch.randint(0, len(OPERATIONS), (batch_size,), device=device)
    lengths = torch.randint(2, max_len - 1, (batch_size,), device=device)
    x = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full_like(x, PAD_ID)
    for task_index, operation in enumerate(OPERATIONS):
        for length in range(2, max_len - 1):
            rows = task_ids.eq(task_index) & lengths.eq(length)
            count = int(rows.sum())
            if not count:
                continue
            if operation == "translate":
                source_is_letters = torch.randint(0, 2, (count, 1), device=device).bool()
                digits = torch.randint(1, 4, (count, length), device=device)
                letters = torch.tensor([LETTER_IDS["a"], LETTER_IDS["b"], LETTER_IDS["c"]], device=device)[digits - 1]
                payload = torch.where(source_is_letters, letters, digits)
            else:
                payload = torch.randint(0, 10, (count, length), device=device)
            x[rows, 0] = OPERATION_IDS[operation]
            x[rows, 1:length + 1], x[rows, length + 1] = payload, 10
            y[rows, 1:length + 1], y[rows, length + 1] = transform(payload, operation), 10
    return x, y


@dataclass
class TrainConfig:
    operation: str = "reverse"
    seq_len: int = 9
    steps: int = 1000
    batch_size: int = 256
    learning_rate: float = 0.003
    d_model: int = 16
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 16
    seed: int = 42


class Trainer:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model: TinyTransformer | None = None
        self.model_name: str | None = None
        self.operation: str | None = None
        self.history: list[dict] = []
        self.status = "idle"
        self.error: str | None = None
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self.listeners: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def state(self) -> dict:
        with self._lock:
            return {
                "status": self.status, "device": str(self.device), "model_name": self.model_name,
                "operation": self.operation, "history": self.history[-1000:], "error": self.error,
            }

    def start(self, cfg: TrainConfig):
        if cfg.operation not in TRAINING_MODES:
            raise ValueError("Unsupported operation")
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Training is already running")
        self.loop = asyncio.get_running_loop()
        self.stop_event.clear()
        self.status, self.error, self.history = "training", None, []
        self._thread = threading.Thread(target=self._train, args=(cfg,), daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()

    def _publish(self, message: dict):
        if self.loop:
            for queue in list(self.listeners):
                self.loop.call_soon_threadsafe(queue.put_nowait, message)

    def _train(self, cfg: TrainConfig):
        try:
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            vocab_size = len(MULTITASK_VOCAB) if cfg.operation in ("multi_task", "translate") else len(VOCAB)
            model_cfg = ModelConfig(seq_len=cfg.seq_len, vocab_size=vocab_size,
                                    d_model=cfg.d_model, n_heads=cfg.n_heads,
                                    n_layers=cfg.n_layers, d_ff=cfg.d_ff)
            model = TinyTransformer(model_cfg).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
            start = time.time()
            for step in range(1, cfg.steps + 1):
                if self.stop_event.is_set():
                    break
                x, y = (make_multitask_batch(cfg.batch_size, cfg.seq_len, self.device)
                        if cfg.operation == "multi_task"
                        else make_batch(cfg.batch_size, cfg.seq_len, cfg.operation, self.device))
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step == 1 or step % 10 == 0 or step == cfg.steps:
                    valid = y.ne(PAD_ID)
                    accuracy = ((logits.argmax(-1) == y) & valid).sum().div(valid.sum()).item()
                    metric = {"step": step, "loss": round(loss.item(), 6),
                              "accuracy": round(accuracy, 6), "seconds": round(time.time()-start, 2)}
                    with self._lock:
                        self.history.append(metric)
                    self._publish({"type": "metric", **metric})
            name = f"{cfg.operation}-{time.strftime('%Y%m%d-%H%M%S')}"
            with self._lock:
                self.model, self.model_name, self.operation = model, name, cfg.operation
                self.status = "stopped" if self.stop_event.is_set() else "complete"
            self.save(name, cfg)
            self._publish({"type": "status", **self.state()})
        except Exception as exc:
            self.status, self.error = "error", str(exc)
            self._publish({"type": "status", **self.state()})

    def save(self, name: str, train_cfg: TrainConfig | None = None):
        if self.model is None:
            return
        payload = {"format_version": 1, "model_config": self.model.cfg.dict(),
                   "train_config": asdict(train_cfg) if train_cfg else None,
                   "operation": self.operation, "history": self.history,
                   "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}
        torch.save(payload, self.checkpoint_dir / f"{name}.pt")

    def checkpoints(self):
        result = []
        for path in sorted(self.checkpoint_dir.glob("*.pt"), reverse=True):
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                result.append({"name": path.stem, "operation": data.get("operation"),
                               "steps": len(data.get("history", [])), "size": path.stat().st_size})
            except Exception:
                result.append({"name": path.stem, "operation": "invalid", "steps": 0, "size": path.stat().st_size})
        return result

    def load(self, name: str):
        path = self.checkpoint_dir / f"{Path(name).name}.pt"
        if not path.exists():
            raise FileNotFoundError(name)
        data = torch.load(path, map_location=self.device, weights_only=False)
        model = TinyTransformer(ModelConfig(**data["model_config"])).to(self.device)
        model.load_state_dict(data["state_dict"])
        model.eval()
        with self._lock:
            self.model, self.model_name = model, path.stem
            self.operation, self.history, self.status = data.get("operation"), data.get("history", []), "loaded"
        return self.state()

    @torch.inference_mode()
    def inspect(self, text: str, operation: str | None = None):
        if self.model is None:
            raise RuntimeError("Train or load a model first")
        encoded = encode(text, self.model.cfg.seq_len - (1 if self.operation == "multi_task" else 0))
        if self.operation == "multi_task":
            operation = operation or "reverse"
            if operation not in OPERATIONS:
                raise ValueError("Choose a valid task for multi-task inspection")
            encoded = torch.cat((torch.tensor([OPERATION_IDS[operation]]), encoded))
        tokens = encoded.unsqueeze(0).to(self.device)
        self.model.eval()
        logits, internals = self.model(tokens, capture=True)
        probs = logits.softmax(-1)
        predicted_tokens = probs.argmax(-1)[0]
        if self.operation == "multi_task":
            predicted_tokens = predicted_tokens[1:]
        prediction = decode(predicted_tokens, MULTITASK_VOCAB if self.operation in ("multi_task", "translate") else VOCAB)
        layers = []
        for item in internals:
            mlp = item["mlp"][0]
            layers.append({
                "attention": item["attention"][0].detach().cpu().tolist(),
                "mlp": mlp.detach().cpu().tolist(),
                "mlp_mean": mlp.mean(0).detach().cpu().tolist(),
            })
        shown_input = text if text.endswith("=") else text + "="
        token_labels = ([OPERATION_TOKENS[operation]] if self.operation == "multi_task" else []) + list(shown_input)
        return {"input": shown_input, "tokens": token_labels, "prediction": prediction,
                "operation": operation if self.operation == "multi_task" else self.operation,
                "confidence": probs.max(-1).values[0].cpu().tolist(), "layers": layers}
