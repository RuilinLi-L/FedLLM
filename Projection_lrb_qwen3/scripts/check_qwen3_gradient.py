#!/usr/bin/env python3
"""Run one local Qwen3 BF16 classification gradient diagnostic; no attack is implemented."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = QWEN_ROOT.parent
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

import torch

from src.gradient_capture import (
    build_canonical_gradient_manifest,
    capture_single_example_gradients,
)
from src.hashing import sha256_json
from src.qwen3_classifier import (
    load_local_qwen3_sequence_classifier,
    tokenize_single_example,
)
from src.result_schema import write_or_verify_json
from src.span_diagnostics import diagnose_two_q_projections


def _resolve_output_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(QWEN_ROOT)
    except ValueError as error:
        raise ValueError(f"Output path must remain under {QWEN_ROOT}, got {resolved}.") from error
    return resolved


def parse_args() -> argparse.Namespace:
    """Parse one strict local single-sample gradient diagnostic request."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one BF16 Qwen3 SST-2-style classification forward/backward and "
            "validate q_proj inputs against raw gradient row spaces. No DAGER or LRB is run."
        )
    )
    parser.add_argument("--sentence", required=True, help="One non-empty SST-2 sentence.")
    parser.add_argument("--label", required=True, type=int, choices=(0, 1), help="Binary class label.")
    parser.add_argument("--head-seed", required=True, type=int, help="Explicit random classification-head seed.")
    parser.add_argument(
        "--model-path",
        default="models/Qwen3-1.7B-Base",
        help="Repository-relative local Qwen3-1.7B-Base path.",
    )
    parser.add_argument("--device", default="cuda", help="Explicit CUDA device, e.g. cuda or cuda:0.")
    parser.add_argument("--max-length", type=int, default=32, help="Maximum unpadded EOS-terminated length.")
    parser.add_argument("--rank-tol", type=float, default=1e-6, help="Absolute FP32 singular-value rank threshold.")
    parser.add_argument(
        "--max-relative-residual",
        type=float,
        default=1e-4,
        help="Largest allowed q_proj input residual relative to the selected raw row space.",
    )
    parser.add_argument(
        "--gradient-orientation",
        choices=("gradient", "gradient.T"),
        default="gradient",
        help="Explicit matrix whose row space is tested; no automatic transpose is attempted.",
    )
    parser.add_argument(
        "--output",
        default="Projection_lrb_qwen3/outputs/gradient_diagnostic.json",
        help="Structured diagnostic JSON output under Projection_lrb_qwen3/.",
    )
    parser.add_argument(
        "--manifest-output",
        default="Projection_lrb_qwen3/outputs/canonical_named_gradient_manifest.json",
        help="Canonical named-gradient manifest JSON output under Projection_lrb_qwen3/.",
    )
    return parser.parse_args()


def _write_error(path: Path, args: argparse.Namespace, error: Exception) -> None:
    """Persist a structured error without concealing the original exception type."""
    identity = {
        "record_type": "qwen3_gradient_diagnostic_error",
        "sentence": args.sentence,
        "label": args.label,
        "head_seed": args.head_seed,
        "model_path": args.model_path,
        "gradient_orientation": args.gradient_orientation,
        "rank_tol": args.rank_tol,
        "max_relative_residual": args.max_relative_residual,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    document = {
        "schema_version": 1,
        "status": "error",
        "diagnostic_sha256": sha256_json(identity),
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    write_or_verify_json(path, document, identity_key="diagnostic_sha256", ignored_existing_keys=("created_at",))


def main() -> int:
    """Execute the diagnostic and fail closed if q inputs miss the selected row space."""
    args = parse_args()
    try:
        output_path = _resolve_output_path(args.output)
        manifest_path = _resolve_output_path(args.manifest_output)
        model_path = (REPOSITORY_ROOT / args.model_path).resolve() if not Path(args.model_path).is_absolute() else Path(args.model_path).resolve()
        target_device = torch.device(args.device)
        if target_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("An available explicit CUDA device is required; no CPU fallback exists.")
        torch.cuda.reset_peak_memory_stats(target_device)
        bundle = load_local_qwen3_sequence_classifier(
            model_path,
            head_seed=args.head_seed,
            device=target_device,
        )
        batch = tokenize_single_example(
            bundle.tokenizer,
            sentence=args.sentence,
            label=args.label,
            max_length=args.max_length,
            device=bundle.device,
        )
        captured = capture_single_example_gradients(
            bundle.model,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        manifest = build_canonical_gradient_manifest(bundle.model)
        manifest_identity = {
            "model_path": args.model_path,
            "head_seed": args.head_seed,
            "entries": manifest["entries"],
            "gradient_tensor_count": manifest["gradient_tensor_count"],
            "gradient_numel": manifest["gradient_numel"],
        }
        manifest_document: dict[str, Any] = {
            "schema_version": 1,
            "record_type": "canonical_named_gradient_manifest",
            "manifest_sha256": sha256_json(manifest_identity),
            "model_path": args.model_path,
            "head_seed": args.head_seed,
            "q_parameter_names": list(captured.q_parameter_names),
            "q_expected_weight_shape": list(captured.q_expected_weight_shape),
            **manifest,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        write_or_verify_json(
            manifest_path,
            manifest_document,
            identity_key="manifest_sha256",
            ignored_existing_keys=("created_at",),
        )
        diagnostics = diagnose_two_q_projections(
            q_inputs=captured.q_inputs,
            q_gradients=captured.q_gradients,
            q_parameter_names=captured.q_parameter_names,
            rank_tol=args.rank_tol,
            max_relative_residual=args.max_relative_residual,
            gradient_orientation=args.gradient_orientation,
        )
        status = "ok" if diagnostics["passed"] else "failed_span_residual"
        diagnostic_identity = {
            "manifest_sha256": manifest_document["manifest_sha256"],
            "sentence": args.sentence,
            "label": args.label,
            "token_ids": list(batch.token_ids),
            "loss": captured.loss,
            "gpu_peak_memory_bytes": captured.gpu_peak_memory_bytes,
            "span_diagnostics": diagnostics,
            "status": status,
        }
        document = {
            "schema_version": 1,
            "record_type": "qwen3_single_sample_gradient_diagnostic",
            "status": status,
            "diagnostic_sha256": sha256_json(diagnostic_identity),
            "manifest_sha256": manifest_document["manifest_sha256"],
            "model_path": args.model_path,
            "head_seed": args.head_seed,
            "sentence": args.sentence,
            "label": args.label,
            "batch_size": 1,
            "padding": False,
            "attention_mask_all_ones": True,
            "explicit_eos": True,
            "token_ids": list(batch.token_ids),
            "loss": captured.loss,
            "gpu_peak_memory_bytes": captured.gpu_peak_memory_bytes,
            "q_expected_weight_shape": list(captured.q_expected_weight_shape),
            "q_gradient_shapes": [list(gradient.shape) for gradient in captured.q_gradients],
            "q_input_shapes": [list(q_input.shape) for q_input in captured.q_inputs],
            "full_gradient_tensor_count": manifest["gradient_tensor_count"],
            "full_gradient_total_elements": manifest["gradient_numel"],
            "span_diagnostics": diagnostics,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        write_or_verify_json(
            output_path,
            document,
            identity_key="diagnostic_sha256",
            ignored_existing_keys=("created_at",),
        )
        print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        return 0 if status == "ok" else 3
    except Exception as error:
        try:
            output_path = _resolve_output_path(args.output)
            _write_error(output_path, args, error)
        except Exception as write_error:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "error_write_failure": str(write_error),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
