"""Verify the public repository manifest and the locked MUSE-IDR ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ONLY_PREFIXES = (
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".venv/",
    "cache/",
    "data/cache/",
    "data/caid_holdout/",
    "data/external/",
    "data/features/",
    "data/interim/",
    "data/processed/",
    "data/raw/",
    "logs/",
    "htmlcov/",
    "mlruns/",
    "outputs/ablations/",
    "outputs/analysis/",
    "reproduced/",
    "third_party/",
    "tmp/",
    "tools/",
    "venv/",
    "wandb/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_public_file(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    parts = path.relative_to(ROOT).parts
    return (
        path.is_file()
        and path != ROOT / "MANIFEST.sha256"
        and ".git" not in parts
        and "__pycache__" not in parts
        and not relative.endswith((".pyc", ".incomplete"))
        and not relative.startswith(LOCAL_ONLY_PREFIXES)
    )


def verify_result_locks(failures: list[str]) -> tuple[int, int]:
    """Verify every nested result lock from the public repository root."""

    lock_files = sorted((ROOT / "results").rglob("*.sha256"))
    entry_count = 0
    for lock_path in lock_files:
        lock_relative = lock_path.relative_to(ROOT).as_posix()
        for line_number, raw_line in enumerate(
            lock_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            if "  " not in raw_line:
                failures.append(
                    f"malformed result lock entry: {lock_relative}:{line_number}"
                )
                continue
            expected_hash, relative_text = raw_line.split("  ", maxsplit=1)
            expected_hash = expected_hash.strip().lower()
            relative_text = relative_text.strip().replace("\\", "/")
            if len(expected_hash) != 64 or any(
                character not in "0123456789abcdef" for character in expected_hash
            ):
                failures.append(
                    f"invalid result lock hash: {lock_relative}:{line_number}"
                )
                continue
            relative_path = Path(relative_text)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                failures.append(
                    f"unsafe result lock path: {lock_relative}:{line_number}: "
                    f"{relative_text}"
                )
                continue
            entry_count += 1
            target_path = ROOT / relative_path
            if not target_path.is_file():
                failures.append(
                    f"missing result lock target: {lock_relative}: {relative_text}"
                )
                continue
            observed_hash = sha256(target_path)
            if observed_hash != expected_hash:
                failures.append(
                    f"result lock hash mismatch: {lock_relative}: {relative_text}"
                )
    return len(lock_files), entry_count


def main() -> None:
    failures: list[str] = []
    manifest_path = ROOT / "MANIFEST.sha256"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    listed_files: set[str] = set()
    for line in lines:
        expected, relative = line.split("  ", 1)
        if relative in listed_files:
            failures.append(f"duplicate manifest entry: {relative}")
        listed_files.add(relative)
        path = ROOT / Path(relative)
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path).lower() != expected.lower():
            failures.append(f"hash mismatch: {relative}")

    actual_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if is_public_file(path)
    }
    extra_files = sorted(actual_files - listed_files)
    missing_manifest_files = sorted(listed_files - actual_files)
    failures.extend(f"unlisted public file: {path}" for path in extra_files)
    failures.extend(
        f"manifest file absent from repository: {path}"
        for path in missing_manifest_files
    )

    result_lock_files, result_lock_entries = verify_result_locks(failures)

    lock_path = (
        ROOT
        / "experiments/final_models/a10_locked_inference/a10_lock_manifest.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    members = lock["checkpoints"]
    family_counts = {
        family: sum(row["family"] == family for row in members)
        for family in ("a2", "a8")
    }
    for member in members:
        relative = Path(member["relative_path"])
        checkpoint = ROOT / "models" / "weights" / "a10_locked" / relative
        if not checkpoint.is_file():
            failures.append(f"missing checkpoint: {relative.as_posix()}")
        elif sha256(checkpoint).lower() != member["sha256"].lower():
            failures.append(f"checkpoint hash mismatch: {relative.as_posix()}")

    development = list((ROOT / "models" / "weights" / "a10_locked").rglob("*development*"))
    status = (
        "pass"
        if not failures
        and len(members) == 30
        and family_counts == {"a2": 15, "a8": 15}
        and not development
        else "fail"
    )
    report = {
        "status": status,
        "manifest_entries": len(lines),
        "result_lock_files": result_lock_files,
        "result_lock_entries": result_lock_entries,
        "locked_members": len(members),
        "family_counts": family_counts,
        "development_checkpoint_excluded": not development,
        "local_only_prefixes_excluded": list(LOCAL_ONLY_PREFIXES),
        "extra_public_files": extra_files,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
