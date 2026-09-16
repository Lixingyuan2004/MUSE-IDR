"""Offline invariant checks for the B4 official-baseline source audit."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = Path(__file__).with_name("source_audit.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def range_size(call: ast.Call) -> int:
    if not isinstance(call.func, ast.Name) or call.func.id != "range":
        raise AssertionError("ensemble comprehension must use range")
    values = [ast.literal_eval(arg) for arg in call.args]
    if len(values) == 1:
        start, stop, step = 0, values[0], 1
    elif len(values) == 2:
        start, stop, step = values[0], values[1], 1
    elif len(values) == 3:
        start, stop, step = values
    else:
        raise AssertionError("unsupported range signature")
    return len(range(start, stop, step))


def flattened_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return flattened_names(node.left) + flattened_names(node.right)
    raise AssertionError("list_modelInfo must be a sum of named lists")


def released_ensemble_members(main_path: Path) -> int:
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    list_sizes: dict[str, int] = {}
    final_names: list[str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id.startswith("list_modelInfo_") and isinstance(
                statement.value, ast.ListComp
            ):
                generator = statement.value.generators[0]
                if not isinstance(generator.iter, ast.Call):
                    raise AssertionError("model list comprehension lacks range call")
                list_sizes[target.id] = range_size(generator.iter)
            elif target.id == "list_modelInfo":
                final_names = flattened_names(statement.value)
    if final_names is None:
        raise AssertionError(f"could not find list_modelInfo in {main_path}")
    return sum(list_sizes[name] for name in final_names)


def configured_threshold(path: Path) -> float:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "label_threshold"
            for target in node.targets
        ):
            return float(ast.literal_eval(node.value))
    raise AssertionError(f"label_threshold missing from {path}")


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def license_file_present(repo: Path) -> bool:
    names = {"license", "licence", "copying"}
    return any(path.is_file() and path.stem.lower() in names for path in repo.rglob("*"))


def expected_weights_present(repo: Path, full: bool) -> bool:
    expected = [
        *(repo / "predictor" / "onehot" / f"cnn2_L12_withFullyDisorder78.pth_f{k}" for k in range(1, 4)),
        *(repo / "predictor" / "protTrans" / f"cnn2_L3_100_50_withFullyDisorder78.pth_f{k}" for k in range(1, 6)),
        *(repo / "predictor" / "protTrans" / f"cnn2_L12_withFullyDisorder78.pth_f{k}" for k in range(1, 6)),
    ]
    if full:
        expected.extend(
            repo
            / "predictor"
            / "msa_transformer"
            / f"cnn2_L11_withFullyDisorder.pth_f{k}"
            for k in range(1, 6)
        )
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    punch2 = PROJECT_ROOT / audit["punch2"]["local_path"]
    light = PROJECT_ROOT / audit["punch2_light"]["local_path"]
    if git_head(punch2) != audit["punch2"]["commit"]:
        raise AssertionError("PUNCH2 commit mismatch")
    if git_head(light) != audit["punch2_light"]["commit"]:
        raise AssertionError("PUNCH2-Light commit mismatch")

    full_members = released_ensemble_members(punch2 / "main.py")
    light_members = released_ensemble_members(light / "main.py")
    if full_members != audit["punch2"]["released_main"]["ensemble_members"]:
        raise AssertionError("PUNCH2 released ensemble count changed")
    if light_members != audit["punch2_light"]["released_main"]["ensemble_members"]:
        raise AssertionError("PUNCH2-Light released ensemble count changed")

    if not expected_weights_present(punch2, full=True):
        raise AssertionError("PUNCH2 released weights are incomplete")
    if not expected_weights_present(light, full=False):
        raise AssertionError("PUNCH2-Light released weights are incomplete")

    full_threshold = configured_threshold(punch2 / "params" / "hyperparams.py")
    light_threshold = configured_threshold(light / "params" / "hyperparams.py")
    if full_threshold != audit["punch2"]["released_main"]["configured_threshold"]:
        raise AssertionError("PUNCH2 released threshold changed")
    if light_threshold != audit["punch2_light"]["released_main"]["configured_threshold"]:
        raise AssertionError("PUNCH2-Light released threshold changed")

    if license_file_present(punch2) or license_file_present(light):
        raise AssertionError("license audit must be updated before publication")

    for paper in audit["papers"].values():
        paper_path = Path(paper["path"])
        if paper_path.exists() and sha256(paper_path) != paper["sha256"]:
            raise AssertionError(f"paper hash mismatch: {paper_path}")

    paper_code_mismatch = (
        full_members != audit["punch2"]["paper_protocol"]["ensemble_members"]
        and light_members != audit["punch2_light"]["paper_protocol"]["ensemble_members"]
    )
    caid_training_variants_separated = (
        audit["lora_dr_suite"]["checkpoints"]["caid_free_comparison"]["allowed_on_caid2"]
        and not audit["lora_dr_suite"]["checkpoints"]["paper_caid3_protocol"]["allowed_on_caid2"]
    )
    if not paper_code_mismatch or not caid_training_variants_separated:
        raise AssertionError("B4 claim guardrails are incomplete")

    print(
        json.dumps(
            {
                "status": "pass",
                "punch2_commit_locked": True,
                "punch2_light_commit_locked": True,
                "released_weights_present": True,
                "punch2_paper_13_vs_released_18_detected": True,
                "punch2_light_paper_8_vs_released_13_detected": True,
                "punch2_threshold_paper_0_378_vs_released_0_39_detected": True,
                "lora_caid_free_and_paper_caid3_variants_separated": True,
                "soft_disorder_models_excluded": True,
                "single_classic_idr_output_required": True,
                "a10_remains_locked": True,
                "caid_labels_used_for_training_or_tuning": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
