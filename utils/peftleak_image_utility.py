from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import EncoderBlock, VisionTransformer

from utils.defense_common import capture_gradients, overwrite_gradients
from utils.defenses import apply_defense
from utils.lrb_presets import apply_lrb_preset


FORMAL_PROFILE = "formal_vit_b16"
DEBUG_PROFILE = "debug_tiny"
PROFILE_CHOICES = (FORMAL_PROFILE, DEBUG_PROFILE)
FORMAL_ADAPTER_TENSOR_COUNT = 96


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.up(self.activation(self.down(inputs)))


class AdapterizedEncoderBlock(nn.Module):
    """Torchvision ViT block with adapters on attention and MLP outputs."""

    def __init__(self, block: EncoderBlock, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.block = block
        self.attn_adapter = BottleneckAdapter(hidden_dim, bottleneck_dim)
        self.mlp_adapter = BottleneckAdapter(hidden_dim, bottleneck_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        torch._assert(inputs.dim() == 3, f"Expected (batch, sequence, hidden), got {inputs.shape}")
        attention_input = self.block.ln_1(inputs)
        attention_output, _ = self.block.self_attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        attention_output = self.block.dropout(attention_output)
        hidden = inputs + self.attn_adapter(attention_output)
        mlp_output = self.block.mlp(self.block.ln_2(hidden))
        return hidden + self.mlp_adapter(mlp_output)


class AdapterizedVisionTransformer(nn.Module):
    def __init__(self, backbone: VisionTransformer, *, bottleneck_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_classes = int(num_classes)
        hidden_dim = int(backbone.hidden_dim)

        wrapped_layers = []
        for block in backbone.encoder.layers:
            if not isinstance(block, EncoderBlock):
                raise TypeError(f"Unsupported torchvision ViT block: {type(block)!r}")
            wrapped_layers.append(AdapterizedEncoderBlock(block, hidden_dim, self.bottleneck_dim))
        backbone.encoder.layers = nn.Sequential(*wrapped_layers)

        if not hasattr(backbone.heads, "head"):
            raise AttributeError("Torchvision ViT does not expose backbone.heads.head.")
        backbone.heads.head = nn.Linear(hidden_dim, self.num_classes)

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if self._is_shared_adapter_name(name) or self._is_local_head_name(name):
                parameter.requires_grad_(True)

    @staticmethod
    def _is_shared_adapter_name(name: str) -> bool:
        return ".attn_adapter." in name or ".mlp_adapter." in name

    @staticmethod
    def _is_local_head_name(name: str) -> bool:
        return name.startswith("backbone.heads.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def shared_parameter_items(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if self._is_shared_adapter_name(name)
        ]

    def shared_parameter_names(self) -> list[str]:
        return [name for name, _ in self.shared_parameter_items()]

    def shared_parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.shared_parameter_items()]

    def local_head_parameter_items(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if self._is_local_head_name(name)
        ]

    def local_head_parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.local_head_parameter_items()]

    def frozen_parameter_items(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, parameter) for name, parameter in self.named_parameters() if not parameter.requires_grad]


def build_utility_model(
    *,
    profile: str,
    num_classes: int = 100,
    bottleneck_dim: int = 64,
    pretrained: bool = True,
) -> AdapterizedVisionTransformer:
    if profile == FORMAL_PROFILE:
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vit_b_16(weights=weights)
    elif profile == DEBUG_PROFILE:
        backbone = VisionTransformer(
            image_size=32,
            patch_size=8,
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            mlp_dim=128,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported image utility profile: {profile!r}")

    model = AdapterizedVisionTransformer(
        backbone,
        bottleneck_dim=bottleneck_dim,
        num_classes=num_classes,
    )
    if profile == FORMAL_PROFILE and len(model.shared_parameters()) != FORMAL_ADAPTER_TENSOR_COUNT:
        raise RuntimeError(
            "Formal ViT-B/16 must expose exactly 96 Adapter tensors; "
            f"got {len(model.shared_parameters())}."
        )
    return model


class SharedAdapterDefenseView:
    def __init__(self, model: AdapterizedVisionTransformer):
        self.model = model

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.model.shared_parameters()

    def trainable_parameter_names(self) -> list[str]:
        return self.model.shared_parameter_names()


def apply_shared_adapter_defense(model: AdapterizedVisionTransformer, args) -> tuple[torch.Tensor | None, ...]:
    shared_parameters = model.shared_parameters()
    raw_gradients = capture_gradients(shared_parameters)
    if any(gradient is None for gradient in raw_gradients):
        missing = [
            name
            for name, gradient in zip(model.shared_parameter_names(), raw_gradients)
            if gradient is None
        ]
        raise RuntimeError(f"Missing shared Adapter gradients: {missing[:5]}")

    reported_defense = str(getattr(args, "defense", "none"))
    if reported_defense in {"proj_only", "full_lrb"}:
        args.defense_lrb_preset = reported_defense
        args.defense = "lrb"
    try:
        apply_lrb_preset(args)
        defended = apply_defense(
            raw_gradients,
            args,
            model_wrapper=SharedAdapterDefenseView(model),
        )
    finally:
        args.defense = reported_defense

    if len(defended) != len(shared_parameters):
        raise RuntimeError("Defended Adapter gradient count changed.")
    for name, parameter, gradient in zip(model.shared_parameter_names(), shared_parameters, defended):
        if gradient is None:
            raise RuntimeError(f"Defense removed required Adapter gradient: {name}")
        if gradient.shape != parameter.shape:
            raise RuntimeError(
                f"Defense changed Adapter gradient shape for {name}: "
                f"{tuple(gradient.shape)} != {tuple(parameter.shape)}"
            )
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"Defense produced non-finite Adapter gradient: {name}")
    overwrite_gradients(shared_parameters, defended)
    return defended


@dataclass(frozen=True)
class ImageDatasetBundle:
    train: torch.utils.data.Dataset
    validation: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    synthetic: bool


def _formal_transforms():
    mean = ViT_B_16_Weights.IMAGENET1K_V1.transforms().mean
    std = ViT_B_16_Weights.IMAGENET1K_V1.transforms().std
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_image_datasets(
    *,
    profile: str,
    data_root: str | Path,
    split_seed: int,
    validation_size: int,
    download: bool,
    debug_train_size: int = 32,
    debug_validation_size: int = 16,
    debug_test_size: int = 16,
) -> ImageDatasetBundle:
    if profile == DEBUG_PROFILE:
        transform = transforms.ToTensor()
        return ImageDatasetBundle(
            train=datasets.FakeData(debug_train_size, (3, 32, 32), 100, transform=transform, random_offset=0),
            validation=datasets.FakeData(
                debug_validation_size,
                (3, 32, 32),
                100,
                transform=transform,
                random_offset=10_000,
            ),
            test=datasets.FakeData(
                debug_test_size,
                (3, 32, 32),
                100,
                transform=transform,
                random_offset=20_000,
            ),
            synthetic=True,
        )
    if profile != FORMAL_PROFILE:
        raise ValueError(f"Unsupported image utility profile: {profile!r}")

    train_transform, eval_transform = _formal_transforms()
    train_source = datasets.CIFAR100(root=str(data_root), train=True, transform=train_transform, download=download)
    validation_source = datasets.CIFAR100(
        root=str(data_root),
        train=True,
        transform=eval_transform,
        download=download,
    )
    test = datasets.CIFAR100(root=str(data_root), train=False, transform=eval_transform, download=download)
    if validation_size <= 0 or validation_size >= len(train_source):
        raise ValueError("validation_size must leave non-empty train and validation splits.")
    generator = torch.Generator().manual_seed(int(split_seed))
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return ImageDatasetBundle(
        train=Subset(train_source, train_indices),
        validation=Subset(validation_source, validation_indices),
        test=test,
        synthetic=False,
    )


def build_image_loaders(
    bundle: ImageDatasetBundle,
    *,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(int(seed))
    common = {"num_workers": int(num_workers), "pin_memory": bool(pin_memory)}
    train_loader = DataLoader(
        bundle.train,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(bundle.validation, batch_size=int(eval_batch_size), shuffle=False, **common)
    test_loader = DataLoader(bundle.test, batch_size=int(eval_batch_size), shuffle=False, **common)
    return train_loader, validation_loader, test_loader


def adapter_checkpoint_state(model: AdapterizedVisionTransformer) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "adapter_state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in model.shared_parameter_items()
        },
        "head_state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in model.local_head_parameter_items()
        },
    }


def load_adapter_checkpoint_state(model: AdapterizedVisionTransformer, state: dict) -> None:
    named_parameters = dict(model.named_parameters())
    for scope in ("adapter_state_dict", "head_state_dict"):
        values = state.get(scope)
        if not isinstance(values, dict):
            raise ValueError(f"Checkpoint is missing {scope}.")
        for name, value in values.items():
            if name not in named_parameters:
                raise ValueError(f"Checkpoint parameter is not present in model: {name}")
            named_parameters[name].data.copy_(value.to(named_parameters[name]))


def validate_parameter_scopes(model: AdapterizedVisionTransformer, *, formal: bool) -> None:
    shared_names = set(model.shared_parameter_names())
    local_names = {name for name, _ in model.local_head_parameter_items()}
    if shared_names & local_names:
        raise RuntimeError("Shared Adapter and local head parameter scopes overlap.")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if trainable_names != shared_names | local_names:
        unexpected = sorted(trainable_names.difference(shared_names | local_names))
        raise RuntimeError(f"Unexpected trainable image parameters: {unexpected[:5]}")
    if formal and len(shared_names) != FORMAL_ADAPTER_TENSOR_COUNT:
        raise RuntimeError(f"Formal shared Adapter scope has {len(shared_names)} tensors, expected 96.")
