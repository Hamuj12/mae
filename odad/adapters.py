"""Lightweight residual adapters for ODAD shadow adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AdapterSpec:
    layer_idx: int
    in_channels: int
    hidden_channels: int
    reduction: int
    min_channels: int
    scale: float
    param_count: int


class ResidualConvAdapter(nn.Module):
    """Bottleneck 1x1 residual adapter initialized as an identity function."""

    def __init__(
        self,
        in_channels: int,
        reduction: int = 8,
        min_channels: int = 8,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        hidden_channels = max(int(min_channels), max(1, in_channels // max(1, int(reduction))))
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.reduction = int(reduction)
        self.min_channels = int(min_channels)
        self.scale = float(scale)
        self.down = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.act = nn.SiLU()
        self.up = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + float(self.scale) * self.up(self.act(self.down(x)))


class MemoryConditionedResidualConvAdapter(ResidualConvAdapter):
    """Residual adapter with optional object-memory conditioning."""

    def __init__(
        self,
        in_channels: int,
        reduction: int = 8,
        min_channels: int = 8,
        scale: float = 1.0,
        memory_dim: int = 32,
        conditioning: str = "film",
        enable_source_projection: bool = False,
        memory_bank_size: int = 16,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            reduction=reduction,
            min_channels=min_channels,
            scale=scale,
        )
        memory_dim = max(1, int(memory_dim))
        conditioning = str(conditioning)
        if conditioning not in {"film", "concat"}:
            raise RuntimeError(f"Unsupported memory adapter conditioning: {conditioning}")
        self.memory_dim = memory_dim
        self.memory_conditioning = conditioning
        self.memory_bank_size = max(1, int(memory_bank_size))
        self.register_buffer("memory_context", torch.zeros(memory_dim), persistent=True)
        self.register_buffer("memory_slots", torch.zeros(self.memory_bank_size, memory_dim), persistent=True)
        self.register_buffer("memory_active_slots", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("memory_initialized", torch.tensor(False), persistent=True)
        self.last_conditioning_norm: float = 0.0

        self.film: Optional[nn.Linear]
        self.memory_to_hidden: Optional[nn.Linear]
        if conditioning == "film":
            self.film = nn.Linear(memory_dim, 2 * self.in_channels, bias=True)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            with torch.no_grad():
                self.film.bias[: self.in_channels].fill_(1.0)
            self.memory_to_hidden = None
        elif conditioning == "concat":
            self.film = None
            self.memory_to_hidden = nn.Linear(memory_dim, self.hidden_channels, bias=True)
            nn.init.zeros_(self.memory_to_hidden.weight)
            nn.init.zeros_(self.memory_to_hidden.bias)

        self.memory_projector: Optional[nn.Linear]
        if bool(enable_source_projection):
            self.memory_projector = nn.Linear(self.in_channels, memory_dim, bias=True)
        else:
            self.memory_projector = None

    def set_memory_context(self, value: torch.Tensor, initialized: bool = True) -> None:
        with torch.no_grad():
            context = value.detach().to(device=self.memory_context.device, dtype=self.memory_context.dtype).flatten()
            if int(context.numel()) != int(self.memory_dim):
                raise RuntimeError(
                    f"Memory context dim mismatch: got {int(context.numel())}, expected {int(self.memory_dim)}."
                )
            self.memory_context.copy_(context)
            self.memory_initialized.fill_(bool(initialized))

    def clear_memory_context(self) -> None:
        with torch.no_grad():
            self.memory_context.zero_()
            self.memory_slots.zero_()
            self.memory_active_slots.zero_()
            self.memory_initialized.fill_(False)
            self.last_conditioning_norm = 0.0

    def set_memory_bank_slots(self, slots: torch.Tensor, active_slots: int, initialized: bool = True) -> None:
        with torch.no_grad():
            active = max(0, min(int(active_slots), int(self.memory_bank_size)))
            self.memory_slots.zero_()
            if active > 0:
                value = slots.detach().to(device=self.memory_slots.device, dtype=self.memory_slots.dtype)
                if value.ndim != 2 or int(value.shape[1]) != int(self.memory_dim):
                    raise RuntimeError(
                        f"Memory slot dim mismatch: got {tuple(value.shape)}, expected [S,{self.memory_dim}]."
                    )
                self.memory_slots[:active].copy_(value[:active])
            self.memory_active_slots.fill_(active)
            self.memory_initialized.fill_(bool(initialized and active > 0))

    def project_memory_feature(self, pooled_feature: torch.Tensor) -> torch.Tensor:
        if self.memory_projector is None:
            raise RuntimeError("Selected memory source adapter does not have a memory projector.")
        feature = pooled_feature.to(device=self.memory_projector.weight.device, dtype=self.memory_projector.weight.dtype)
        if feature.ndim != 1 or int(feature.numel()) != int(self.in_channels):
            raise RuntimeError(
                f"Expected pooled feature shape [{self.in_channels}], got {tuple(feature.shape)}."
            )
        return self.memory_projector(feature)

    def memory_conditioning_norm(self) -> float:
        return float(self.last_conditioning_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.down(x)
        if bool(self.memory_initialized.item()) and self.memory_conditioning == "concat":
            assert self.memory_to_hidden is not None
            mem = self.memory_context.to(device=x.device, dtype=x.dtype)
            hidden = hidden + self.memory_to_hidden(mem).view(1, -1, 1, 1)
        residual = self.up(self.act(hidden))
        if bool(self.memory_initialized.item()) and self.memory_conditioning == "film":
            assert self.film is not None
            mem = self.memory_context.to(device=x.device, dtype=x.dtype)
            film = self.film(mem)
            gamma, beta = film.chunk(2, dim=0)
            residual = gamma.view(1, -1, 1, 1) * residual + beta.view(1, -1, 1, 1)
            self.last_conditioning_norm = float(
                torch.cat([(gamma.detach() - 1.0).flatten(), beta.detach().flatten()]).float().norm().cpu()
            )
        elif bool(self.memory_initialized.item()) and self.memory_conditioning == "concat":
            self.last_conditioning_norm = float(residual.detach().float().norm().cpu())
        else:
            self.last_conditioning_norm = 0.0
        return x + float(self.scale) * residual


class AdaptedLayer(nn.Module):
    """Wrap a YOLO layer and apply a residual adapter to its tensor output."""

    def __init__(self, base: nn.Module, adapter: ResidualConvAdapter) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.f = getattr(base, "f", -1)
        self.i = getattr(base, "i", -1)
        base_type = str(getattr(base, "type", base.__class__.__name__))
        self.type = f"{base_type}.Adapter"
        self.np = int(getattr(base, "np", 0)) + sum(param.numel() for param in adapter.parameters())

    def forward(self, x):
        out = self.base(x)
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(
                f"AdaptedLayer expects tensor output at YOLO layer {self.i}, got {type(out).__name__}."
            )
        return self.adapter(out)


def parse_adapter_layers(value: str) -> List[int]:
    layers: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(int(item))
    if not layers:
        raise RuntimeError("At least one adapter layer index is required.")
    seen: Set[int] = set()
    ordered: List[int] = []
    for idx in layers:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def _layer_sequence(core_model: nn.Module) -> nn.Module:
    layer_seq = getattr(core_model, "model", None)
    if not isinstance(layer_seq, (nn.ModuleList, nn.Sequential)):
        raise RuntimeError("Expected YOLO core model to expose a mutable ModuleList/Sequential at .model.")
    return layer_seq


def _infer_output_channels(
    core_model: nn.Module,
    layers: Sequence[nn.Module],
    selected_layers: Sequence[int],
    imgsz: int,
    device: str,
) -> Dict[int, int]:
    selected = set(int(idx) for idx in selected_layers)
    if not selected:
        return {}
    max_selected = max(selected)
    was_training = bool(core_model.training)
    channels: Dict[int, int] = {}
    x = torch.zeros(1, 3, int(imgsz), int(imgsz), device=device)
    saved: List[Optional[torch.Tensor]] = []
    core_model.eval()
    with torch.no_grad():
        for idx, module in enumerate(layers):
            from_idx = getattr(module, "f", -1)
            if from_idx != -1:
                if isinstance(from_idx, int):
                    x = saved[from_idx]
                else:
                    x = [x if int(j) == -1 else saved[int(j)] for j in from_idx]
            out = module(x)
            if idx in selected:
                if not isinstance(out, torch.Tensor) or out.ndim != 4:
                    raise RuntimeError(
                        f"Adapter layer {idx} must produce a BCHW tensor, got {type(out).__name__}."
                    )
                channels[idx] = int(out.shape[1])
            save_indices = getattr(core_model, "save", [])
            saved.append(out if getattr(module, "i", idx) in save_indices else None)
            x = out
            if idx >= max_selected:
                break
    core_model.train(was_training)
    return channels


def attach_residual_adapters(
    core_model: nn.Module,
    layers: Sequence[nn.Module],
    adapter_layers: Sequence[int],
    imgsz: int,
    device: str,
    reduction: int,
    min_channels: int,
    scale: float,
    memory_enable: bool = False,
    memory_dim: int = 32,
    memory_conditioning: str = "film",
    memory_source_layer: Optional[int] = None,
    memory_bank_size: int = 16,
) -> Tuple[List[nn.Module], List[AdapterSpec]]:
    layer_seq = _layer_sequence(core_model)
    n_layers = len(layers)
    selected = [int(idx) for idx in adapter_layers]
    new_selected: List[int] = []
    for idx in selected:
        if idx < 0 or idx >= n_layers:
            raise RuntimeError(f"Invalid adapter layer {idx}; expected range [0, {n_layers - 1}].")
        base_module = layers[idx].base if isinstance(layers[idx], AdaptedLayer) else layers[idx]
        if base_module.__class__.__name__ in {"Detect", "Segment"}:
            raise RuntimeError(f"Adapter layer {idx} is a prediction head; choose a feature layer before Detect.")
        if not isinstance(layers[idx], AdaptedLayer):
            new_selected.append(idx)

    channels = _infer_output_channels(
        core_model=core_model,
        layers=layers,
        selected_layers=new_selected,
        imgsz=int(imgsz),
        device=str(device),
    )
    specs: List[AdapterSpec] = []
    for idx in selected:
        if isinstance(layer_seq[idx], AdaptedLayer):
            adapter = layer_seq[idx].adapter
            specs.append(
                AdapterSpec(
                    layer_idx=idx,
                    in_channels=int(adapter.in_channels),
                    hidden_channels=int(adapter.hidden_channels),
                    reduction=int(adapter.reduction),
                    min_channels=int(adapter.min_channels),
                    scale=float(adapter.scale),
                    param_count=int(sum(param.numel() for param in adapter.parameters())),
                )
            )
            continue
        if bool(memory_enable):
            adapter = MemoryConditionedResidualConvAdapter(
                in_channels=channels[idx],
                reduction=int(reduction),
                min_channels=int(min_channels),
                scale=float(scale),
                memory_dim=int(memory_dim),
                conditioning=str(memory_conditioning),
                enable_source_projection=(memory_source_layer is not None and int(idx) == int(memory_source_layer)),
                memory_bank_size=int(memory_bank_size),
            )
        else:
            adapter = ResidualConvAdapter(
                in_channels=channels[idx],
                reduction=int(reduction),
                min_channels=int(min_channels),
                scale=float(scale),
            )
        wrapper = AdaptedLayer(layer_seq[idx], adapter)
        layer_seq[idx] = wrapper
        specs.append(
            AdapterSpec(
                layer_idx=idx,
                in_channels=int(adapter.in_channels),
                hidden_channels=int(adapter.hidden_channels),
                reduction=int(adapter.reduction),
                min_channels=int(adapter.min_channels),
                scale=float(adapter.scale),
                param_count=int(sum(param.numel() for param in adapter.parameters())),
            )
        )
    return list(layer_seq), specs


def adapter_modules(model: nn.Module) -> List[ResidualConvAdapter]:
    return [module for module in model.modules() if isinstance(module, ResidualConvAdapter)]


def memory_adapter_modules(model: nn.Module) -> List[MemoryConditionedResidualConvAdapter]:
    return [module for module in model.modules() if isinstance(module, MemoryConditionedResidualConvAdapter)]


def set_memory_context(model: nn.Module, value: torch.Tensor, initialized: bool = True) -> None:
    for adapter in memory_adapter_modules(model):
        adapter.set_memory_context(value, initialized=initialized)


def set_memory_bank_slots(model: nn.Module, slots: torch.Tensor, active_slots: int, initialized: bool = True) -> None:
    for adapter in memory_adapter_modules(model):
        adapter.set_memory_bank_slots(slots, active_slots=active_slots, initialized=initialized)


def sync_memory_context_from_student(teacher_model: nn.Module, student_model: nn.Module) -> None:
    teacher_adapters = memory_adapter_modules(teacher_model)
    student_adapters = memory_adapter_modules(student_model)
    if len(teacher_adapters) != len(student_adapters):
        return
    for teacher_adapter, student_adapter in zip(teacher_adapters, student_adapters):
        teacher_adapter.set_memory_context(
            student_adapter.memory_context.detach(),
            initialized=bool(student_adapter.memory_initialized.item()),
        )
        if hasattr(teacher_adapter, "set_memory_bank_slots") and hasattr(student_adapter, "memory_slots"):
            teacher_adapter.set_memory_bank_slots(
                student_adapter.memory_slots.detach(),
                active_slots=int(student_adapter.memory_active_slots.item()),
                initialized=bool(student_adapter.memory_initialized.item()),
            )


def memory_adapter_param_count(model: nn.Module) -> int:
    total = 0
    for adapter in memory_adapter_modules(model):
        for name, param in adapter.named_parameters():
            if (
                name.startswith("film.")
                or name.startswith("memory_to_hidden.")
                or name.startswith("memory_projector.")
            ):
                total += int(param.numel())
    return int(total)


def memory_adapter_stats(model: nn.Module) -> Dict[str, float]:
    adapters = memory_adapter_modules(model)
    initialized = any(bool(adapter.memory_initialized.item()) for adapter in adapters)
    norms = [float(adapter.memory_context.detach().float().norm().cpu()) for adapter in adapters]
    cond_norms = [float(adapter.memory_conditioning_norm()) for adapter in adapters]

    return {
        "memory_adapter_initialized": float(int(initialized)),
        "memory_adapter_mean_norm": float(sum(norms) / len(norms)) if norms else float("nan"),
        "mean_memory_conditioning_norm": float(sum(cond_norms) / len(cond_norms)) if cond_norms else float("nan"),
    }


def adapter_param_count(model: nn.Module) -> int:
    return int(sum(param.numel() for adapter in adapter_modules(model) for param in adapter.parameters()))


def adapter_param_id_set(model: nn.Module) -> Set[int]:
    return {id(param) for adapter in adapter_modules(model) for param in adapter.parameters()}


def detect_head_param_id_set(layers: Sequence[nn.Module], head_idx: int) -> Set[int]:
    return {id(param) for param in layers[int(head_idx)].parameters()}


def adapter_state_keys(model: nn.Module) -> Set[str]:
    return {key for key in model.state_dict().keys() if ".adapter." in key}


def assert_matching_adapter_state(student_model: nn.Module, teacher_model: nn.Module) -> None:
    student_keys = adapter_state_keys(student_model)
    teacher_keys = adapter_state_keys(teacher_model)
    if student_keys != teacher_keys:
        missing_teacher = sorted(student_keys - teacher_keys)
        missing_student = sorted(teacher_keys - student_keys)
        raise RuntimeError(
            "Student/teacher adapter state mismatch: "
            f"missing_teacher={missing_teacher[:5]} missing_student={missing_student[:5]}"
        )


def apply_adapter_freeze_policy(
    model: nn.Module,
    layers: Sequence[nn.Module],
    head_idx: int,
    train_detect_head: bool,
) -> Set[int]:
    for param in model.parameters():
        param.requires_grad = False
    expected = adapter_param_id_set(model)
    for param in model.parameters():
        if id(param) in expected:
            param.requires_grad = True
    if bool(train_detect_head):
        head_ids = detect_head_param_id_set(layers, int(head_idx))
        expected.update(head_ids)
        for param in layers[int(head_idx)].parameters():
            param.requires_grad = True
    return expected


def freeze_frozen_batchnorm_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(
            param.requires_grad for param in module.parameters(recurse=False)
        ):
            module.eval()


def adapter_debug_stats(model: nn.Module, previous: Optional[Mapping[str, torch.Tensor]] = None) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if ".adapter." not in name:
            continue
        value = param.detach()
        stats[f"{name}.norm"] = float(value.norm().cpu())
        if previous is not None and name in previous:
            stats[f"{name}.delta_norm"] = float((value.cpu() - previous[name]).norm())
    return stats


def snapshot_adapter_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if ".adapter." in name
    }
