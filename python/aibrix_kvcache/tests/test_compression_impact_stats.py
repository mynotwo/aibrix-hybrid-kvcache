"""Independent tests for paired benchmark statistics."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[1] / "tools" / "compression_impact_stats.py"
SPEC = importlib.util.spec_from_file_location("compression_impact_stats", TOOL)
stats = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stats
SPEC.loader.exec_module(stats)


def test_missing_metrics_and_ci_crossing_zero_are_inconclusive():
    rows = [
        {
            "OFF": {
                "valid": True,
                "throughput_tokens_per_second": 100.0,
                "requests": [
                    {"valid": True, "ttft_seconds": 1.0, "tpot_seconds": None}
                ],
            },
            "LIGHT": {
                "valid": True,
                "throughput_tokens_per_second": rate,
                "requests": [
                    {"valid": True, "ttft_seconds": 1.0, "tpot_seconds": None}
                ],
            },
        }
        for rate in (98.0, 102.0) * 15
    ]
    report = stats.summarize_pairs(rows, bootstrap_samples=200, seed=4)

    assert (
        report["metrics"]["throughput_tokens_per_second"]["status"]
        == "INCONCLUSIVE"
    )
    assert report["metrics"]["tpot_seconds"]["delta_percent"] is None
    assert report["metrics"]["tpot_seconds"]["ci95"] is None


def test_invalid_trace_cannot_enter_paired_statistics():
    with pytest.raises(ValueError, match="invalid trace"):
        stats.summarize_pairs(
            [
                {
                    "OFF": {"valid": False, "requests": []},
                    "LIGHT": {"valid": True, "requests": []},
                }
            ]
        )


def test_pairing_rejects_unbalanced_group_lengths():
    rows = [{"token_ids": [1], "tokens_per_second": 1}]
    with pytest.raises(ValueError, match="same repetition count"):
        stats.paired_rows(
            [rows, rows, rows, []], ["OFF", "LIGHT", "LIGHT", "OFF"]
        )


def test_p99_is_bootstrapped_by_paired_batch_but_is_not_a_gate():
    rows = []
    for off, light in ((1.0, 1.01), (2.0, 2.02), (3.0, 3.03)):
        rows.append(
            {
                "OFF": {
                    "valid": True,
                    "throughput_tokens_per_second": 10,
                    "requests": [
                        {
                            "valid": True,
                            "ttft_seconds": off,
                            "tpot_seconds": off,
                        }
                    ],
                },
                "LIGHT": {
                    "valid": True,
                    "throughput_tokens_per_second": 10,
                    "requests": [
                        {
                            "valid": True,
                            "ttft_seconds": light,
                            "tpot_seconds": light,
                        }
                    ],
                },
            }
        )
    report = stats.summarize_pairs(rows, bootstrap_samples=100, seed=8)
    p99 = report["metrics"]["p99_ttft_seconds"]
    assert p99["delta_percent"] is not None
    assert p99["ci95"] is not None
    assert p99["status"] == "DESCRIPTIVE"
