"""Download pinned PDBe quality metadata for S2 candidate structure mappings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
API_BASE = "https://www.ebi.ac.uk/pdbe/api/v2"
REQUEST_LIST_SHA256 = "55beb2fbe1dca55bfb81b5fdbcfbfdaef8f10b2a37ad6ba02f39c73a8016447f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    values = [line.strip().lower() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("duplicate PDB identifiers in request list")
    if any(len(value) != 4 or not value.isalnum() for value in values):
        raise ValueError("invalid PDB identifier in request list")
    return sorted(values)


def request_json(url: str, *, body: object | None = None, attempts: int = 5) -> object:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Accept": "application/json",
        "User-Agent": "MUSE-IDR S2 evidence audit",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET"
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 16))
    raise AssertionError("unreachable")


def batched_metadata(pdb_ids: list[str], endpoint: str, batch_size: int) -> dict[str, object]:
    merged: dict[str, object] = {}
    for start in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[start:start + batch_size]
        response = request_json(f"{API_BASE}{endpoint}", body=",".join(batch))
        if not isinstance(response, dict):
            raise ValueError(f"unexpected PDBe response for {endpoint}")
        normalized = {str(key).lower(): value for key, value in response.items()}
        unexpected = set(normalized) - set(batch)
        duplicate = set(merged) & set(normalized)
        if unexpected or duplicate:
            raise ValueError(
                f"invalid PDBe response for {endpoint}: unexpected={sorted(unexpected)}, "
                f"duplicate={sorted(duplicate)}"
            )
        merged.update(normalized)
        print(
            json.dumps(
                {
                    "endpoint": endpoint,
                    "completed": min(start + batch_size, len(pdb_ids)),
                    "total": len(pdb_ids),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request-list",
        type=Path,
        default=Path(
            "outputs/supplementary/s2_independent_holdout/evidence_audit_20260906/"
            "pdb_metadata_request_ids.txt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/pdbe/2026-09-06/s2_candidate_pdb_metadata.json"),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request_list = args.request_list if args.request_list.is_absolute() else ROOT / args.request_list
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pinned metadata: {output}")
    if sha256(request_list) != REQUEST_LIST_SHA256:
        raise ValueError("PDB request list SHA256 mismatch")
    if args.batch_size < 1 or args.batch_size > 100:
        raise ValueError("batch size must be in [1, 100]")
    pdb_ids = read_ids(request_list)
    summary = batched_metadata(pdb_ids, "/pdb/entry/summary", args.batch_size)
    experiment = batched_metadata(pdb_ids, "/pdb/entry/experiment", args.batch_size)
    openapi = request_json(f"{API_BASE}/openapi.json")
    payload = {
        "schema_version": 1,
        "purpose": "s2_independent_holdout_structure_quality_audit",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "api_base": API_BASE,
        "endpoints": {
            "summary": "/pdb/entry/summary",
            "experiment": "/pdb/entry/experiment",
            "openapi": "/openapi.json",
        },
        "request_list": request_list.relative_to(ROOT).as_posix(),
        "request_list_sha256": sha256(request_list),
        "requested_pdb_ids": pdb_ids,
        "summary": summary,
        "experiment": experiment,
        "missing_summary_ids": sorted(set(pdb_ids) - set(summary)),
        "missing_experiment_ids": sorted(set(pdb_ids) - set(experiment)),
        "openapi": openapi,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": "pass",
                "requested": len(pdb_ids),
                "summary": len(summary),
                "experiment": len(experiment),
                "missing_summary": len(payload["missing_summary_ids"]),
                "missing_experiment": len(payload["missing_experiment_ids"]),
                "output": output.relative_to(ROOT).as_posix(),
                "output_sha256": sha256(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
