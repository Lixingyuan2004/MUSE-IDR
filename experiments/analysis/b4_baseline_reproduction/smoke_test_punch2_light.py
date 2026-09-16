from __future__ import annotations

import json

import numpy as np
import torch

from punch2_light_common import (
    DEFAULT_PUNCH2_LIGHT_REPO,
    author_onehot,
    ensemble_probabilities,
    load_official_members,
    paper8_member_ids,
    predict_member_probabilities,
    released_member_specs,
    validate_feature_array,
    verify_repository,
)


def main() -> None:
    repo = DEFAULT_PUNCH2_LIGHT_REPO
    revision = verify_repository(repo)
    specs = released_member_specs(repo)
    if len(specs) != 13 or len(paper8_member_ids()) != 8:
        raise AssertionError("PUNCH2-Light member counts are incorrect")

    sequence = "ACDUZOBJX"
    onehot = author_onehot(sequence)
    if onehot.shape != (1, len(sequence), 21):
        raise AssertionError("official one-hot shape mismatch")
    x_index = 20
    if not all(onehot[0, position, x_index] == 1 for position in (3, 4, 5, 6)):
        raise AssertionError("official ambiguous-residue mapping mismatch")
    if np.any(onehot[0, 7]):
        raise AssertionError("J should preserve the author's all-zero one-hot behavior")

    device = torch.device("cpu")
    specs, models, hashes = load_official_members(repo, device)
    generator = np.random.default_rng(20260826)
    prottrans = generator.normal(0, 0.1, size=(1, len(sequence), 1024)).astype(np.float32)
    onehot = validate_feature_array(onehot, len(sequence), 21, "onehot")
    prottrans = validate_feature_array(prottrans, len(sequence), 1024, "protTrans")
    members = predict_member_probabilities(
        specs,
        models,
        onehot,
        prottrans,
        device,
    )
    paper8 = ensemble_probabilities(members, "paper8")
    released13 = ensemble_probabilities(members, "released13")
    manual8 = np.stack(
        [members[key] for key in sorted(paper8_member_ids())],
        axis=0,
    ).mean(axis=0)
    manual13 = np.stack([members[key] for key in sorted(members)], axis=0).mean(axis=0)
    if not np.allclose(paper8, manual8, atol=0, rtol=0):
        raise AssertionError("paper8 uniform ensemble mismatch")
    if not np.allclose(released13, manual13, atol=0, rtol=0):
        raise AssertionError("released13 uniform ensemble mismatch")
    if paper8.shape != (len(sequence),) or released13.shape != (len(sequence),):
        raise AssertionError("one-score-per-residue invariant failed")
    if not all(hash_value and len(hash_value) == 64 for hash_value in hashes.values()):
        raise AssertionError("checkpoint hash audit failed")

    result = {
        "status": "pass",
        "source_revision": revision,
        "official_checkpoints_loaded": len(models),
        "paper_faithful_members": 8,
        "released_entry_point_members": 13,
        "paper8_uniform_probability_mean_correct": True,
        "released13_uniform_probability_mean_correct": True,
        "official_uzob_to_x_mapping": True,
        "official_j_zero_onehot_behavior_preserved": True,
        "one_score_per_residue": True,
        "output_heads": 1,
        "soft_disorder_output": False,
        "caid_labels_accessed": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
