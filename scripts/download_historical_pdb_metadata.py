"""Download PDBe metadata for PDB IDs cited by retained historical DisProt records."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from muse_idr.data.pdb_negatives import parse_historical_pdb_cross_reference


PD_BE_API = "https://www.ebi.ac.uk/pdbe/api/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--disprot-json",
        type=Path,
        default=Path(
            "data/caid_holdout/references/caid1/rebuild_sources/disprot-2018-11.json"
        ),
    )
    parser.add_argument(
        "--candidate-jsonl",
        type=Path,
        default=Path("data/interim/historical_disprot_2018/classic_idr_candidates.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/pdbe/2026-08-21/historical_disprot_pdb_metadata.json"),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def retained_disprot_ids(candidate_jsonl: Path) -> set[str]:
    identifiers: set[str] = set()
    with candidate_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            values = row.get("disprot_ids")
            if not isinstance(values, list):
                raise ValueError(f"Missing disprot_ids at {candidate_jsonl}:{line_number}")
            identifiers.update(str(value).upper() for value in values)
    if not identifiers:
        raise ValueError(f"No retained DisProt IDs found in {candidate_jsonl}")
    return identifiers


def cited_pdb_ids(
    disprot_json: Path, retained_ids: set[str]
) -> tuple[set[str], list[dict[str, str]]]:
    payload = json.loads(disprot_json.read_text(encoding="utf-8"))
    entries = payload.get("data")
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected DisProt JSON structure: {disprot_json}")
    pdb_ids: set[str] = set()
    invalid_cross_references: list[dict[str, str]] = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or str(entry.get("disprot_id", "")).upper() not in retained_ids
        ):
            continue
        regions = entry.get("regions", [])
        if not isinstance(regions, list):
            raise ValueError(f"Malformed regions in {entry.get('disprot_id')}")
        for region in regions:
            if not isinstance(region, dict):
                continue
            cross_refs = region.get("cross_refs", []) or []
            for cross_ref in cross_refs:
                if not isinstance(cross_ref, dict):
                    continue
                if str(cross_ref.get("db", "")).strip().upper() == "PDB":
                    raw_pdb_id = str(cross_ref.get("id", "")).strip()
                    parsed_ids = parse_historical_pdb_cross_reference(raw_pdb_id)
                    if not parsed_ids:
                        invalid_cross_references.append(
                            {
                                "disprot_id": str(entry.get("disprot_id", "")).upper(),
                                "region_id": str(region.get("region_id", "")),
                                "raw_value": raw_pdb_id,
                            }
                        )
                        continue
                    pdb_ids.update(parsed_ids)
    if not pdb_ids:
        raise ValueError("No historical PDB cross-references found")
    return pdb_ids, invalid_cross_references


def request_json(url: str, *, body: object | None = None, attempts: int = 4) -> object:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Accept": "application/json",
        "User-Agent": "MUSE-IDR research data builder",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data else "GET",
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("unreachable")


def batched_metadata(pdb_ids: list[str], endpoint: str, batch_size: int) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    merged: dict[str, object] = {}
    for start in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[start : start + batch_size]
        response = request_json(f"{PD_BE_API}{endpoint}", body=",".join(batch))
        if not isinstance(response, dict):
            raise ValueError(f"Unexpected PDBe response for {endpoint}")
        duplicate = merged.keys() & response.keys()
        if duplicate:
            raise ValueError(f"Duplicate PDBe response IDs: {sorted(duplicate)}")
        merged.update({str(key).lower(): value for key, value in response.items()})
        print(f"{endpoint}: {min(start + batch_size, len(pdb_ids))}/{len(pdb_ids)}")
    return merged


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    disprot_json = _resolve(root, args.disprot_json)
    candidate_jsonl = _resolve(root, args.candidate_jsonl)
    output = _resolve(root, args.output)
    retained_ids = retained_disprot_ids(candidate_jsonl)
    cited_ids, invalid_cross_references = cited_pdb_ids(disprot_json, retained_ids)
    pdb_ids = sorted(cited_ids)
    summary = batched_metadata(pdb_ids, "/pdb/entry/summary", args.batch_size)
    experiment = batched_metadata(pdb_ids, "/pdb/entry/experiment", args.batch_size)
    openapi = request_json(f"{PD_BE_API}/openapi.json")
    payload = {
        "schema_version": 1,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "api_base": PD_BE_API,
        "endpoints": {
            "summary": "/pdb/entry/summary",
            "experiment": "/pdb/entry/experiment",
            "openapi": "/openapi.json",
        },
        "retained_disprot_ids": len(retained_ids),
        "requested_pdb_ids": pdb_ids,
        "invalid_historical_cross_references": invalid_cross_references,
        "summary": summary,
        "experiment": experiment,
        "missing_summary_ids": sorted(set(pdb_ids) - summary.keys()),
        "missing_experiment_ids": sorted(set(pdb_ids) - experiment.keys()),
        "openapi": openapi,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "requested": len(pdb_ids),
                "summary": len(summary),
                "experiment": len(experiment),
                "missing_summary": len(payload["missing_summary_ids"]),
                "missing_experiment": len(payload["missing_experiment_ids"]),
                "invalid_historical_cross_references": len(invalid_cross_references),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
