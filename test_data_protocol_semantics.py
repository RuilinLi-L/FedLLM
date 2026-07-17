#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import utils.data as data_module


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, (list, np.ndarray)):
            return {name: [self.rows[int(idx)][name] for idx in key] for name in self.rows[0]}
        return self.rows[int(key)]


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def _fake_load_dataset(*args, **kwargs):
    train = FakeDataset({"sentence": f"train-{idx}", "label": idx % 2} for idx in range(1200))
    validation = FakeDataset({"sentence": f"validation-{idx}", "label": idx % 2} for idx in range(300))
    return {"train": train, "validation": validation}


def test_official_validation_uses_official_partition_without_replacement():
    original = data_module.load_dataset
    data_module.load_dataset = _fake_load_dataset
    try:
        np.random.seed(101)
        dataset = data_module.TextDataset("cpu", "sst2", "official_validation", 10, 2)
    finally:
        data_module.load_dataset = original

    flattened = [text for batch in dataset.seqs for text in batch]
    assert_true(dataset.data_protocol == "official_validation", "official protocol should be recorded")
    assert_true(dataset.source_partition == "validation", "official source partition should be recorded")
    assert_true(all(text.startswith("validation-") for text in flattened), "private samples must come from validation")
    assert_true(len(flattened) == len(set(flattened)), "official validation sampling must be without replacement")


def test_legacy_internal_protocol_remains_on_training_partition():
    original = data_module.load_dataset
    data_module.load_dataset = _fake_load_dataset
    try:
        np.random.seed(101)
        dataset = data_module.TextDataset("cpu", "sst2", "test", 2, 2)
    finally:
        data_module.load_dataset = original

    assert_true(dataset.data_protocol == "legacy_internal", "legacy split must keep its protocol label")
    assert_true(dataset.source_partition == "train", "legacy split must keep using the training partition")
    assert_true(all(text.startswith("train-") for batch in dataset.seqs for text in batch), "legacy samples must remain unchanged")


def _legacy_reference_sequences(split: str, n_inputs: int, batch_size: int) -> list[list[str]]:
    full = _fake_load_dataset()["train"]
    idxs = list(range(len(full)))
    np.random.shuffle(idxs)
    n_samples = n_inputs * batch_size
    if split == "test":
        idxs = idxs[:n_samples]
    else:
        idxs = idxs[1000:]
        final_idxs = []
        while len(final_idxs) < n_samples:
            zipped = sorted(((idx, len(full[idx]["sentence"])) for idx in idxs), key=lambda item: item[1])
            chunk_size = max(len(zipped) // n_samples, 1)
            length = min(len(zipped), n_samples - len(final_idxs))
            for pos in range(length):
                offset = chunk_size * pos + np.random.randint(0, chunk_size)
                final_idxs.append(zipped[offset][0])
            np.random.shuffle(idxs)
        np.random.shuffle(final_idxs)
        idxs = final_idxs
    return [
        [full[idxs[i * batch_size + j]]["sentence"] for j in range(batch_size)]
        for i in range(n_inputs)
    ]


def test_legacy_fixed_seed_indices_match_preexisting_reference_algorithm():
    original = data_module.load_dataset
    data_module.load_dataset = _fake_load_dataset
    try:
        for split in ("test", "val"):
            np.random.seed(101)
            actual = data_module.TextDataset("cpu", "sst2", split, 3, 2).seqs
            np.random.seed(101)
            expected = _legacy_reference_sequences(split, 3, 2)
            assert_true(actual == expected, f"legacy {split} fixed-seed indices must remain unchanged")
    finally:
        data_module.load_dataset = original


def test_dataset_summary_fields_are_backward_safe():
    assert_true(
        dict(data_module.dataset_summary_fields(SimpleNamespace())) == {
            "data_protocol": "n/a",
            "source_partition": "n/a",
        },
        "missing protocol metadata should remain parseable",
    )


def main():
    tests = [
        test_official_validation_uses_official_partition_without_replacement,
        test_legacy_internal_protocol_remains_on_training_partition,
        test_legacy_fixed_seed_indices_match_preexisting_reference_algorithm,
        test_dataset_summary_fields_are_backward_safe,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All data protocol semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
