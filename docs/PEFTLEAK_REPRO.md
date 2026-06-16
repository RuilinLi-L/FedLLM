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
