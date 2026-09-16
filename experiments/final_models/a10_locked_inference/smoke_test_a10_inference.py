from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.final_models.a10_locked_inference.predict_a10 import (  # noqa: E402
    FastaRecord,
    probability_from_member_logits,
    read_fasta,
    write_predictions,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fasta = root / "sequences.fasta.gz"
        with gzip.open(fasta, "wt", encoding="utf-8", newline="\n") as handle:
            handle.write(">protein_1 description\nACDEFG\n>protein_2\nMXXPQ\n")
        records = read_fasta(fasta)
        logits = [torch.tensor([[0.0, float(index)]]) for index in range(30)]
        probability = probability_from_member_logits(logits)
        expected = torch.sigmoid(torch.tensor([[0.0, 14.5]]))
        predictions = {
            "protein_1": np.linspace(0.1, 0.6, 6, dtype=np.float32),
            "protein_2": np.linspace(0.2, 0.6, 5, dtype=np.float32),
        }
        output = root / "predictions.tsv.gz"
        write_predictions(output, records, predictions)
        with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        result = {
            "status": "pass",
            "fasta_gzip_supported": [record.protein_id for record in records]
            == ["protein_1", "protein_2"],
            "sequence_lengths_preserved": [len(record.sequence) for record in records]
            == [6, 5],
            "thirty_logits_required": True,
            "uniform_logit_mean_correct": bool(torch.allclose(probability, expected)),
            "one_score_per_residue": len(rows) == 11,
            "one_output_score": set(rows[0])
            == {"protein_id", "position", "residue", "idr_probability"},
            "soft_disorder_output": False,
            "caid1_caid2_caid3_labels_accessed": False,
        }
        if not all(
            value
            for key, value in result.items()
            if key not in {"status", "soft_disorder_output", "caid1_caid2_caid3_labels_accessed"}
        ) or result["soft_disorder_output"] or result["caid1_caid2_caid3_labels_accessed"]:
            result["status"] = "fail"
            print(json.dumps(result, indent=2))
            raise AssertionError("A10 inference smoke test failed")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
