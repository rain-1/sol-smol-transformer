from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .training import OPERATIONS, PRESETS, TRAINING_MODES, TrainConfig, Trainer
from .benchmarks import benchmark_report

ROOT = Path(__file__).resolve().parent.parent
trainer = Trainer(ROOT / "checkpoints")
app = FastAPI(title="Smol Transformer Lab")


class InspectRequest(BaseModel):
    text: str
    operation: str | None = None


@app.get("/api/info")
def info():
    return {"operations": TRAINING_MODES, "inspect_operations": OPERATIONS,
            "presets": PRESETS, **trainer.state()}


@app.get("/api/benchmarks")
def benchmarks():
    return benchmark_report(ROOT / "pareto-results.json")


@app.post("/api/train")
async def train(config: TrainConfig):
    try:
        trainer.start(config)
        return trainer.state()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/stop")
def stop():
    trainer.stop()
    return {"ok": True}


@app.get("/api/checkpoints")
def checkpoints():
    return trainer.checkpoints()


@app.post("/api/checkpoints/{name}/load")
def load_checkpoint(name: str):
    try:
        return trainer.load(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Checkpoint not found") from exc
    except Exception as exc:
        raise HTTPException(400, f"Could not load checkpoint: {exc}") from exc


@app.post("/api/inspect")
def inspect(request: InspectRequest):
    try:
        return trainer.inspect(request.text, request.operation)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.websocket("/api/events")
async def events(websocket: WebSocket):
    await websocket.accept()
    queue = __import__("asyncio").Queue()
    trainer.listeners.add(queue)
    try:
        await websocket.send_json({"type": "status", **trainer.state()})
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        trainer.listeners.discard(queue)


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
