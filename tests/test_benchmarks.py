"""The scaling benchmark stays runnable; its numbers only mean something at real sizes."""

import json

import pytest
from benchmarks import store_scaling


@pytest.mark.parametrize("backend", ["memory", "lancedb"])
def test_store_scaling_benchmark_runs_end_to_end(tmp_path, capsys, backend):
    if backend == "lancedb":
        pytest.importorskip("lancedb")
    output = tmp_path / "report.json"
    argv = ["--backend", backend, "--sizes", "300", "--dim", "16", "--queries", "4", "--ingest", "24"]
    assert store_scaling.main([*argv, "--writes", "8", "--output", str(output), "--workdir", str(tmp_path)]) == 0

    (result,) = json.loads(output.read_text())["results"]
    ingest = result["ingest"]
    assert result["size"] == 300 and ingest["inserted"] + ingest["merged"] == 24 and ingest["merged"] > 0
    recalls = [result["search"][name]["recall_at_10"] for name in store_scaling.FILTERS]
    assert all(recall is None or 0 <= recall <= 1 for recall in recalls)
    if backend == "memory":
        assert set(recalls) <= {1.0, None}, "the exact store must agree with brute force"
    assert result["reads"]["during_ingest"]["reads"] >= 1
    assert "| 300 |" in capsys.readouterr().out
