"""Exact fixed-rule tests for Qwen3 calibration parameter selection."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_selection import select_calibration_configuration
from src.hashing import sha256_json


SAMPLES = ("a" * 64, "b" * 64)


def _rows(
    parameters: dict[str, object],
    *,
    token: tuple[float, float],
    exact: tuple[bool, bool],
    empty: tuple[bool, bool],
    times: tuple[float, float],
    statuses: tuple[str, str] = ("ok", "ok"),
) -> list[dict[str, object]]:
    candidate_id = sha256_json(parameters)
    return [
        {
            "candidate_id": candidate_id,
            "parameters": parameters,
            "sample_key": sample_key,
            "result_status": status,
            "token_recovery": recovery,
            "exact_recovery": recovered_exact,
            "empty_reconstruction": is_empty,
            "attack_time_seconds": attack_time,
        }
        for sample_key, recovery, recovered_exact, is_empty, attack_time, status in zip(
            SAMPLES, token, exact, empty, times, statuses
        )
    ]


class CalibrationSelectionRuleTest(unittest.TestCase):
    def test_mean_token_recovery_is_the_first_priority(self) -> None:
        token_winner = {"tau1": 0.1}
        lower_token = {"tau1": 0.2}
        rows = []
        rows += _rows(token_winner, token=(0.8, 0.8), exact=(False, False), empty=(True, True), times=(99.0, 99.0))
        rows += _rows(lower_token, token=(0.7, 0.7), exact=(True, True), empty=(False, False), times=(0.1, 0.1))
        selected = select_calibration_configuration(rows, expected_sample_keys=SAMPLES).selected
        self.assertEqual(selected.parameters, token_winner)

    def test_exact_recovery_breaks_a_token_recovery_tie(self) -> None:
        lower_exact = {"tau1": 0.1}
        higher_exact = {"tau1": 0.2}
        rows = _rows(lower_exact, token=(0.8, 0.8), exact=(False, False), empty=(False, False), times=(0.1, 0.1))
        rows += _rows(higher_exact, token=(0.8, 0.8), exact=(True, True), empty=(True, True), times=(99.0, 99.0))
        selected = select_calibration_configuration(rows, expected_sample_keys=SAMPLES).selected
        self.assertEqual(selected.parameters, higher_exact)

    def test_empty_rate_then_time_break_the_remaining_ties(self) -> None:
        more_empty = {"tau1": 0.1}
        slower = {"tau1": 0.2}
        faster = {"tau1": 0.3}
        rows = _rows(more_empty, token=(0.8, 0.8), exact=(False, False), empty=(True, True), times=(0.1, 0.1))
        rows += _rows(slower, token=(0.8, 0.8), exact=(False, False), empty=(False, False), times=(2.0, 2.0))
        rows += _rows(faster, token=(0.8, 0.8), exact=(False, False), empty=(False, False), times=(1.0, 1.0))
        selected = select_calibration_configuration(rows, expected_sample_keys=SAMPLES).selected
        self.assertEqual(selected.parameters, faster)

    def test_failed_rows_are_retained_and_scored_as_declared_worst_case(self) -> None:
        robust = {"tau1": 0.1}
        failed = {"tau1": 0.2}
        rows = []
        rows += _rows(robust, token=(0.6, 0.6), exact=(False, False), empty=(False, False), times=(2.0, 2.0))
        rows += _rows(
            failed,
            token=(1.0, 1.0),
            exact=(True, True),
            empty=(False, False),
            times=(0.1, 0.1),
            statuses=("error", "ok"),
        )
        selected = select_calibration_configuration(rows, expected_sample_keys=SAMPLES).selected
        self.assertEqual(selected.parameters, robust)
        failed_summary = next(
            item for item in select_calibration_configuration(rows, expected_sample_keys=SAMPLES).candidates
            if item.parameters == failed
        )
        self.assertEqual(failed_summary.failed_row_count, 1)
        self.assertEqual(failed_summary.mean_token_recovery, 0.5)

    def test_final_tie_uses_canonical_parameter_dictionary_lexicographic_order(self) -> None:
        late = {"tau1": 0.2, "tau2": 0.1}
        early = {"tau1": 0.1, "tau2": 0.2}
        rows = _rows(late, token=(0.5, 0.5), exact=(False, False), empty=(False, False), times=(1.0, 1.0))
        rows += _rows(early, token=(0.5, 0.5), exact=(False, False), empty=(False, False), times=(1.0, 1.0))
        selected = select_calibration_configuration(rows, expected_sample_keys=SAMPLES).selected
        self.assertEqual(selected.parameters, early)


if __name__ == "__main__":
    unittest.main()
