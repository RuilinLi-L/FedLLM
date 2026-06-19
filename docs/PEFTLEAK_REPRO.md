# PEFTLeak Reproduction Notes

This repository contains the PEFTLeak image-side reproduction path under `attacks/peftleak_image/` and `attack_peftleak_image.py`.

Scope:

- `--mode vit_adapter` is the reportable reproduction mode.
- `--mode synthetic_ratio` is a fast semantic/debug path for the ratio recovery kernel.
- Public patch statistics come from a separate public split or `--public_stats_path`, not from attacked samples.
- The core recovery uses autograd adapter gradients from a torchvision ViT backbone with a malicious adapter branch.
- Deterministic patch clustering and reassembly are used before image metrics are computed.

Why this is image/adapter-specific:

- PEFTLeak targets adapter-style image models where patch information can be exposed through adapter gradients.
- The reconstruction math depends on the adapter structure; it is not the same as the FedLLM PEFT text attack path in `attack_peftleak.py`.

Citation:

- PEFTLeak: CVPR 2025, `info-ucr/PEFTLeak`.

Implementation notes:

- `synthetic_ratio` still validates the ratio kernel directly.
- `vit_adapter` should be used for reportable experiments.
- The image entrypoint accepts `--model_path`, `--finetuned_path`, and `--cache_dir` in the same server-style way as the DAGER attack entrypoints.
- The image entrypoint also accepts `--device`; the default `cuda` reuses the repository GPU selector and falls back to CPU if CUDA is unavailable.
- Runtime tensors stay on the selected device through gradient generation, defense, recovery, clustering, and metric computation. Public-stat files are still serialized on CPU for portability.

## Module I/O

| Stage | Input | Output | Role |
| --- | --- | --- | --- |
| Image loading | CIFAR100 or synthetic images | `images`, `labels` on `--device` | Builds the attacked local batch. |
| Public statistics | Public split or `--public_stats_path` | `PatchStatistics(mean, std, ...)` | Estimates patch normalization from non-attacked data. |
| Patch extraction | `images` | `[batch, n_patches, patch_dim]` | Exposes the per-patch vectors targeted by PEFTLeak. |
| Malicious ViT adapter | images, labels, public stats | adapter gradient tuple and names | Produces PEFT/adapter gradients through autograd. |
| Defense hook | adapter gradients | defended adapter gradients | Applies `none/noise/dpsgd/topk/compression/lrb` after gradient generation and before recovery. |
| Patch-space baselines | patches | defended images | Applies approximate `soteria` and `mixup` before gradient generation; these are coverage baselines, not full original-method reproductions. |
| Gradient recovery | defended `weight`/`bias` gradient pairs | recovered patches | Uses the PEFTLeak ratio formula and public-stat denormalization. |
| Reassembly and metrics | recovered/reference patches | reconstructed images, MSE, PSNR, patch recovery rate | Deterministic post-processing for image-side privacy reporting. |

## Result Fields

- `mse` and `psnr`: image-level reconstruction metrics after deterministic patch reassembly.
- `patch_recovery_count` / `patch_count` / `patch_recovery_rate`: exact patch recovery under `--patch_recovery_mse_threshold`.
- `vit_adapter_loss`: cross-entropy loss of the malicious ViT-adapter forward pass in `vit_adapter` mode.
- `batch_top1_acc`: top-1 accuracy of that forward pass on the attacked batch. It is a local utility sanity metric, not a complete downstream PEFT training score.
- `device`: resolved runtime device, for example `cuda:0` or `cpu`.
