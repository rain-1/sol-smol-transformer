import json

from app.benchmarks import benchmark_report


def test_benchmark_thresholds(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"device": "cpu", "runs": [
        {"operation": "copy", "parameters": 10, "d_model": 4, "n_heads": 1,
         "n_layers": 1, "d_ff": 4, "exact_accuracy": .98, "seconds": 1},
        {"operation": "copy", "parameters": 20, "d_model": 8, "n_heads": 1,
         "n_layers": 1, "d_ff": 8, "exact_accuracy": 1, "seconds": 2},
    ]}))
    report = benchmark_report(path)
    assert report["thresholds"]["copy"]["smallest"]["parameters"] == 20
    assert report["thresholds"]["copy"]["class_minima"]["d_model"] == 8
