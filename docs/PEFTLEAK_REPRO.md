# PEFTLeak Reproduction Notes

This repository contains the PEFTLeak-style image-side shared-bin mechanism path under `attacks/peftleak_image/` and `attack_peftleak_image.py`.

Scope:

- `--mode vit_adapter` is the PEFTLeak-style shared-bin mechanism mode.
- `--mode official_vit_adapter` is an official-aligned v1 mechanism path with repeated adapter-gradient probes, deterministic public-CDF bins, and selectable non-oracle grouping.
- `--mode synthetic_ratio` is a fast semantic/debug path for the ratio recovery kernel.
- Public patch statistics come from a separate public split or `--public_stats_path`, not from attacked samples.
- The core recovery uses autograd adapter gradients from a torchvision ViT backbone with a malicious adapter branch.
- Deterministic patch clustering and reassembly are used before image metrics are computed.
- `vit_adapter` is not a faithful, full official PEFTLeak reproduction of the complete malicious ViT/adapter parameter-design pipeline.
- `official_vit_adapter` narrows part of that gap, but remains a v1 alignment layer rather than a byte-for-byte port of `info-ucr/PEFTLeak`; it still uses an external malicious adapter branch instead of a full in-block ViT adapter overwrite.

Why this is image/adapter-specific:

- PEFTLeak targets adapter-style image models where patch information can be exposed through adapter gradients.
- The reconstruction math depends on the adapter structure; it is not the same as the FedLLM PEFT text attack path in `attack_peftleak.py`.

Citation:

- PEFTLeak: CVPR 2025, `info-ucr/PEFTLeak`.

Implementation notes:

- `synthetic_ratio` still validates the ratio kernel directly.
- `vit_adapter` should be used for shared-bin mechanism experiments.
- The image entrypoint accepts `--model_path`, `--finetuned_path`, and `--cache_dir` in the same server-style way as the DAGER attack entrypoints.
- The image entrypoint also accepts `--device`; the default `cuda` reuses the repository GPU selector and falls back to CPU if CUDA is unavailable.
- Add `--fail_on_synthetic_fallback` for runs that must use real CIFAR100 data. Without it, missing CIFAR100 falls back to synthetic smoke data and marks `synthetic_fallback=1` in the summary.
- Runtime tensors stay on the selected device through gradient generation, defense, recovery, clustering, and metric computation. Public-stat files are still serialized on CPU for portability.
- `--n_images` is the total number of attacked images; `--batch_size` controls the client batch size used for each attack call. The summary reports `effective_batch_size` and `attack_batch_count`.
- `--peftleak_profile lightweight|official_cifar32|custom` controls the geometry/probe defaults. `lightweight` preserves smoke-friendly defaults; `official_cifar32` sets CIFAR32-style official parameters (`image_size=32`, `patch_size=16`, `num_bins=320`, adapter dims `64`); `custom` only records that a custom setting is being used.
- Explicit CLI parameters override profile defaults. Summary fields `official_like_config`, `profile_override_count`, and `config_warning` record whether the final configuration still matches the official-style profile.
- `--official_grouping tag|cluster|oracle_debug` changes the official-v1 recovery path. `oracle_debug` is for semantic checks only and should not be used in main privacy tables.
- `--metrics` controls the primary reported metric fields for `mse`, `psnr`, `ssim`, `lpips`, and `patch_recovery`. LPIPS is optional: missing dependencies produce `lpips=n/a` and `lpips_status=unavailable`, not a failed run.

Reportable shared-bin mechanism command:

```bash
python attack_peftleak_image.py --mode vit_adapter \
  --dataset cifar100 \
  --data_root ./models_cache \
  --model_path torchvision_vit_small \
  --fail_on_synthetic_fallback
```

Synthetic fallback runs are useful for smoke tests and mechanism checks only. Do not report them as CIFAR100 PEFTLeak reproduction results.

Official-style v1 smoke command:

```bash
python attack_peftleak_image.py --mode official_vit_adapter \
  --peftleak_profile official_cifar32 \
  --dataset synthetic \
  --n_images 1 \
  --batch_size 1 \
  --public_split_size 4 \
  --adapter_layers first_n \
  --defense none
```

For real CIFAR100 reporting, add `--dataset cifar100 --data_root ./models_cache --fail_on_synthetic_fallback` and do not mix fallback runs into result tables.

## Official PEFTLeak Alignment

| Official component | Local counterpart | Covered now | Not covered yet |
| --- | --- | --- | --- |
| `Design_Model_Adapter.py` | `OfficialAlignedVitWithAdapters` and `OfficialAlignedPatchAdapter` | Repeated adapter-gradient exposure, adjacent-row ratio structure, tag dimension for grouping | Exact official ViT internal parameter overwrite and true MSA/MLP adapter insertion |
| `processing_v2.py` | `build_official_aligned_probe_statistics` | Public patch statistics and deterministic public-CDF bin intervals | Byte-for-byte official preprocessing |
| `recover_adapter.py` | `recover_patches_from_official_adapter_grads` | Adjacent-row gradient-difference recovery, position/public-stat inversion, tag/cluster/oracle-debug grouping switches | Full official post-processing parity and ViT patch-embedding pseudo-inverse |
| `Clustering.py` | `cluster_and_reassemble` plus official tag grouping | Optional constrained k-means when available, deterministic fallback, and tag-based direct grouping | Full official clustering/post-processing parity |

## Module I/O

| Stage | Input | Output | Role |
| --- | --- | --- | --- |
| Image loading | CIFAR10/CIFAR100/TinyImageNet/ImageNet or synthetic smoke images | `images`, `labels` on `--device` | Builds the attacked local batch. |
| Public statistics | Public split or `--public_stats_path` | `PatchStatistics(mean, std, ...)` | Estimates patch normalization from non-attacked data. |
| Patch extraction | `images` | `[batch, n_patches, patch_dim]` | Exposes the per-patch vectors targeted by PEFTLeak. |
| Malicious ViT adapter | images, labels, public stats | adapter gradient tuple and names | Produces PEFT-style shared-bin adapter gradients through autograd; v1 is an external branch, not full internal ViT block surgery. |
| Defense hook | adapter gradients | defended adapter gradients | Applies `none/noise/dpsgd/topk/compression/lrb` after gradient generation and before recovery. |
| Patch-space baselines | patches | defended images | Applies approximate `soteria` and `mixup` before gradient generation; these are coverage baselines, not full original-method reproductions. |
| Gradient recovery | defended `weight`/`bias` gradient pairs | recovered patches | Uses the PEFTLeak ratio formula and public-stat denormalization. |
| Reassembly and metrics | recovered/reference patches | reconstructed images, MSE, PSNR, patch recovery rate | Deterministic post-processing for image-side privacy reporting. |

## Result Fields

- `mse` and `psnr`: primary image-level reconstruction metrics; `primary_metric_source` states whether they come from `direct`, `clustered`, or `n/a`.
- `reproduction_level`: current implementation level, either `peftleak_style_shared_bins`, `peftleak_official_aligned_v1`, or `synthetic_ratio_debug`.
- `peftleak_profile`, `official_like_config`, `profile_override_count`, `config_warning`: configuration provenance fields for lightweight/custom/official-style runs.
- `non_oracle_primary_only`: always `true` for reportable primary metrics. `oracle_*` fields are emitted only for sanity/debug checks.
- `cluster_method`: `constrained_kmeans` when `k_means_constrained` is installed, otherwise `deterministic`.
- `synthetic_fallback` / `fallback_reason`: whether CIFAR100 was unavailable and synthetic smoke data was used.
- `patch_recovery_count` / `patch_count` / `patch_recovery_rate`: exact patch recovery under `--patch_recovery_mse_threshold`.
- `candidate_patch_count`, `nonzero_slot_count`, `ambiguous_position_count`, `empty_position_count`: observable shared-bin recovery metadata.
- `oracle_*`: debug-only fields that use private slot assignments. Do not use these in main privacy tables.
- `vit_adapter_loss`: cross-entropy loss of the malicious ViT-adapter forward pass in `vit_adapter` mode.
- `batch_top1_acc`: top-1 accuracy of that forward pass on the attacked batch. It is a local utility sanity metric, not a complete downstream PEFT training score.

Official-aligned v1 adds:

- `official_alignment_version`: currently `v1`.
- `adapter_layer_count`, `adapter_bottleneck_dim`, `attack_rounds`, `non_oracle_grouping`, `attack_batch_count`.
- `ssim`, `lpips`, and `lpips_status`; `lpips_status` is `ok`, `unavailable`, `not_requested`, or a failure tag.
- `device`: resolved runtime device, for example `cuda:0` or `cpu`.

Recommended AAAI wording:

- Use: "PEFTLeak-style image-side adapter leakage with an official-aligned v1 configuration."
- Use: "Projection-LRB is evaluated on a PEFTLeak-style image/adapter leakage mechanism as a cross-modal supplementary study."
- Avoid: "full reproduction of PEFTLeak" or "byte-for-byte official PEFTLeak implementation."
- Keep `oracle_*` fields out of main privacy tables; use them only to debug whether the constructed probes can recover the intended private slot assignments.
