from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.final_models.a10_locked_inference.prepare_caid_references import (  # noqa: E402
    ReferenceRecord,
    parse_reference,
    write_fasta,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference = root / "reference.fasta"
        reference.write_text(
            ">p1\nACDX\n01--\n>p2\nMJOU\n1100\n",
            encoding="utf-8",
            newline="\n",
        )
        records = parse_reference(reference)
        output = root / "sequences.fasta"
        write_fasta(output, records)
        text = output.read_text(encoding="utf-8")
        result = {
            "status": "pass",
            "three_line_reference_validated": len(records) == 2,
            "ambiguous_residues_supported": records[1].sequence == "MJOU",
            "sequence_label_alignment_validated": all(
                len(record.sequence) == len(record.labels) for record in records
            ),
            "sequence_output_contains_no_labels": "01--" not in text and "1100" not in text,
            "single_output_fasta_written": output.is_file(),
        }
        if not all(value for key, value in result.items() if key != "status"):
            result["status"] = "fail"
            print(json.dumps(result, indent=2))
            raise AssertionError("CAID preparation smoke test failed")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
