"""Original PEFTLeak-style image adapter leakage helpers."""

from .core import (
    PatchStatistics,
    TorchvisionVitWithMaliciousAdapter,
    VitAdapterGradientResult,
    build_public_patch_statistics,
    build_vit_adapter_gradients,
    cluster_and_reassemble,
    design_malicious_adapter_parameters,
    load_public_patch_statistics,
    optimize_patch_baseline,
    recover_patch_from_adapter_grads,
    recover_patches_from_batch,
    recover_patches_from_named_adapter_grads,
    save_public_patch_statistics,
)