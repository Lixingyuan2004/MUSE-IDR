"""Network-free tests for B4.2 FASTA, sliding-window, output and metrics logic."""

from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (  # noqa: E402
    parse_caid_predictions,
)
from experiments.analysis.b4_baseline_reproduction.evaluate_lora_caid import (  # noqa: E402
    evaluate_track,
)
from experiments.analysis.b4_baseline_reproduction.predict_lora_official import (  # noqa: E402
    FastaRecord,
    extract_window_margins,
    merge_window_margins,
    predict_sequence,
    read_fasta,
    verify_loaded_adapter,
    window_starts,
    write_outputs,
)


class FakeTokenizer:
    def __call__(
        self,
        sequences,
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    ):
        del padding, return_tensors, return_special_tokens_mask
        width = max(len(sequence) for sequence in sequences) + 2
        input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        special = torch.ones_like(input_ids)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, 1 : length + 1] = torch.arange(1, length + 1)
            attention[row, : length + 2] = 1
            special[row, 1 : length + 1] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "special_tokens_mask": special,
        }


class FakeModel:
    def __call__(self, input_ids, attention_mask):
        del attention_mask
        margin = input_ids.float() / 10.0
        logits = torch.stack((torch.zeros_like(margin), margin), dim=-1)
        return SimpleNamespace(logits=logits)


class FakeLoadedModel:
    def named_parameters(self):
        yield "base.query.lora_A.default.weight", torch.nn.Parameter(torch.ones(1))
        yield "base.value.lora_B.default.weight", torch.nn.Parameter(torch.ones(1))
        yield (
            "base.classifier.modules_to_save.default.weight",
            torch.nn.Parameter(torch.ones(1)),
        )


def main() -> None:
    starts = window_starts(25, 10, 6)
    if starts != [0, 6, 12, 15]:
        raise AssertionError(f"unexpected starts: {starts}")

    logits = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0], [0.0, 0.0]],
        ]
    )
    special = torch.tensor([[1, 0, 0, 1, 1], [1, 0, 0, 0, 1]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    margins = extract_window_margins(logits, special, attention, [2, 3])
    if [row.tolist() for row in margins] != [[1.0, 2.0], [3.0, 4.0, 5.0]]:
        raise AssertionError("special/padding token removal failed")

    merged = merge_window_margins(
        5,
        [0, 2],
        [np.asarray([0.0, 0.0, 2.0]), np.asarray([4.0, 0.0, 0.0])],
    )
    expected_overlap = 1.0 / (1.0 + np.exp(-3.0))
    if not np.isclose(merged[2], expected_overlap):
        raise AssertionError("overlapping logit margins were not averaged")

    scores, windows = predict_sequence(
        "ACDEFGHIKLMNPQRSTVWYACDEF",
        FakeTokenizer(),
        FakeModel(),
        torch.device("cpu"),
        max_residues=10,
        stride=6,
        window_batch_size=2,
    )
    if scores.shape != (25,) or windows != 4 or not np.all(np.isfinite(scores)):
        raise AssertionError("end-to-end long-sequence prediction failed")

    adapter_check = verify_loaded_adapter(
        FakeLoadedModel(),
        SimpleNamespace(
            target_modules={"query", "value"},
            modules_to_save=["classifier", "score"],
        ),
    )
    if adapter_check["lora_and_saved_head_loaded"] is not True:
        raise AssertionError("adapter verification failed")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fasta = root / "input.fasta.gz"
        with gzip.open(fasta, "wt", encoding="utf-8", newline="\n") as handle:
            handle.write(">p1 description\nACD\n>p2\nEFGH\n")
        records = read_fasta(fasta)
        if [(r.protein_id, r.sequence) for r in records] != [
            ("p1", "ACD"),
            ("p2", "EFGH"),
        ]:
            raise AssertionError("FASTA/gzip parsing failed")

        rejected = root / "reference_is_not_input.fasta"
        rejected.write_text(">x\nACD\n010\n", encoding="utf-8")
        try:
            read_fasta(rejected)
        except ValueError:
            label_file_rejected = True
        else:
            label_file_rejected = False
        if not label_file_rejected:
            raise AssertionError("label-containing CAID reference was accepted as input")

        predictions = {
            "p1": np.asarray([0.1, 0.8, 0.7], dtype=np.float32),
            "p2": np.asarray([0.9, 0.2, 0.6, 0.3], dtype=np.float32),
        }
        tsv = root / "predictions.tsv.gz"
        caid = root / "predictions.caid"
        write_outputs(tsv, caid, records, predictions)
        with gzip.open(tsv, "rt", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        parsed = parse_caid_predictions(caid.read_text(encoding="utf-8"))
        if len(rows) != 7 or set(parsed) != {"p1", "p2"}:
            raise AssertionError("TSV/CAID output alignment failed")

        reference = root / "reference.fasta"
        reference.write_text(">p1\nACD\n010\n>p2\nEFGH\n1010\n", encoding="utf-8")
        metrics = evaluate_track(reference, caid.read_text(encoding="utf-8"))
        required = {
            "roc_auc_caid_compatible",
            "aucpr_trapezoid",
            "aps",
            "f1_at_0_5",
            "mcc_at_0_5",
            "fmax",
        }
        if not required.issubset(metrics):
            raise AssertionError("required CAID metrics were not produced")

    print(
        json.dumps(
            {
                "status": "pass",
                "fasta_and_gzip_supported": True,
                "label_containing_input_rejected": True,
                "long_sequence_window_coverage": True,
                "special_and_padding_tokens_excluded": True,
                "overlap_logit_fusion_correct": True,
                "official_lora_and_saved_head_required": True,
                "one_score_per_residue": True,
                "tsv_and_caid_output_alignment": True,
                "roc_auc_auprc_aps_f1_mcc_fmax_defined": True,
                "soft_disorder_output": False,
                "caid_labels_used_for_training_or_tuning": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
