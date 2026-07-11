from __future__ import annotations

import json
from pathlib import Path


def benchmark_report(path: Path, target: float = 0.99) -> dict:
    if not path.exists():
        return {"available": False, "runs": [], "thresholds": {}}
    data = json.loads(path.read_text())
    operations = sorted({r["operation"] for r in data["runs"]})
    thresholds = {}
    for operation in operations:
        runs = [r for r in data["runs"] if r["operation"] == operation]
        passing = [r for r in runs if r["exact_accuracy"] >= target]
        thresholds[operation] = {
            "target": target,
            "passing_runs": len(passing),
            "smallest": min(passing, key=lambda r: r["parameters"]) if passing else None,
            "class_minima": ({key: min(r[key] for r in passing)
                              for key in ("parameters", "d_model", "n_heads", "n_layers", "d_ff")}
                             if passing else None),
        }
    return {"available": True, "device": data.get("device"), "runs": data["runs"],
            "frontier": data.get("frontier", {}), "thresholds": thresholds}
