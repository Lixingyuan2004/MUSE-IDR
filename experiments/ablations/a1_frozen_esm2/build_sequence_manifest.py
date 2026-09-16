"""Strip labels from an A1 manifest before frozen representation extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_sequence_manifest(source: Path, output: Path) -> dict[str, object]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            source_row = json.loads(raw_line)
            protein_id = str(source_row["id"])
            sequence = str(source_row["sequence"]).upper().replace(" ", "")
            fold = str(source_row["fold"])
            if not protein_id or protein_id in seen:
                raise ValueError(f"missing or duplicate id at {source}:{line_number}")
            if not sequence:
                raise ValueError(f"empty sequence for {protein_id}")
            seen.add(protein_id)
            rows.append({"id": protein_id, "sequence": sequence, "fold": fold})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    return {
        "schema_version": 1,
        "source": source.as_posix(),
        "source_sha256": file_sha256(source),
        "output": output.as_posix(),
        "output_sha256": file_sha256(output),
        "proteins": len(rows),
        "fields": ["id", "sequence", "fold"],
        "contains_residue_labels": False,
        "caid2_caid3_labels_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_sequence_manifest(args.source, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
