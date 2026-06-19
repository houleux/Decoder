from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "parameters": dict(self.parameters),
        }


class MethodRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError("Method name must not be empty")
        self._factories[key] = factory

    def get(self, name: str) -> Callable[..., Any]:
        key = str(name).strip()
        try:
            return self._factories[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown method '{name}'. Available methods: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._factories)


METHOD_REGISTRY = MethodRegistry()


def register_method(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        METHOD_REGISTRY.register(name, factory)
        return factory

    return decorator


def get_method(name: str) -> Callable[..., Any]:
    return METHOD_REGISTRY.get(name)


def list_methods() -> list[str]:
    return METHOD_REGISTRY.names()


METHOD_CATALOG: tuple[MethodSpec, ...] = (
    MethodSpec(name="flooding", family="baseline"),
    MethodSpec(name="random", family="baseline"),
    MethodSpec(name="round_robin", family="baseline"),
    MethodSpec(name="reldec", family="tabular", parameters={"reward": "mean_neighbor_sign"}),
    MethodSpec(name="reldec_misq_local", family="tabular", parameters={"reward": "misq_local"}),
    MethodSpec(name="reldec_misq_global", family="tabular", parameters={"reward": "misq_global"}),
    MethodSpec(name="rel_delta", family="tabular", parameters={"reward": "reldec_delta"}),
    MethodSpec(name="deep_reldec_z1", family="deep", parameters={"z": 1}),
    MethodSpec(name="deep_reldec_zx", family="deep", parameters={"z": "dynamic"}),
    MethodSpec(name="mi_naive_zx", family="mi_naive", parameters={"z": "dynamic"}),
    MethodSpec(name="mi_dqn_zx", family="mi_dqn", parameters={"z": "dynamic"}),
    MethodSpec(name="mi_tabular_zx", family="mi_tabular", parameters={"z": "dynamic"}),
    MethodSpec(name="augmented_max_avg_zx", family="augmented", parameters={"z": "dynamic", "mode": "max_avg"}),
    MethodSpec(name="augmented_max_zx", family="augmented", parameters={"z": "dynamic", "mode": "max"}),
    MethodSpec(name="augmented_average_zx", family="augmented", parameters={"z": "dynamic", "mode": "average"}),
    MethodSpec(name="tabular_augmented_max_avg_zx", family="tabular_augmented", parameters={"z": "dynamic", "mode": "max_avg"}),
    MethodSpec(name="tabular_augmented_max_zx", family="tabular_augmented", parameters={"z": "dynamic", "mode": "max"}),
    MethodSpec(name="tabular_augmented_average_zx", family="tabular_augmented", parameters={"z": "dynamic", "mode": "average"}),
    # Dyna-Q variants (model-based planning, tabular)
    MethodSpec(name="dyna_reldec", family="dyna", parameters={"reward": "mean_neighbor_sign", "z": "dynamic"}),
    MethodSpec(name="dyna_reldelta", family="dyna", parameters={"reward": "reldec_delta", "z": "dynamic"}),
    MethodSpec(name="dyna_mi", family="dyna", parameters={"reward": "mi_local", "z": "dynamic"}),
    MethodSpec(name="dyna_midelta", family="dyna", parameters={"reward": "mi_delta_local", "z": "dynamic"}),
)


TRAINING_POLICY_CATALOG: tuple[MethodSpec, ...] = (
    MethodSpec(name="tabular", family="tabular", parameters={"reward": "mean_neighbor_sign"}),
    MethodSpec(name="reldec_misq_local", family="tabular", parameters={"reward": "misq_local"}),
    MethodSpec(name="reldec_misq_global", family="tabular", parameters={"reward": "misq_global"}),
    MethodSpec(name="rel_delta", family="tabular", parameters={"reward": "reldec_delta"}),
    MethodSpec(name="deep_z1", family="deep", parameters={"z": 1}),
    MethodSpec(name="mi_tabular_zx", family="mi_tabular", parameters={"z": "dynamic"}),
    MethodSpec(name="deep_zx", family="deep", parameters={"z": "dynamic"}),
    MethodSpec(name="mi_dqn_zx", family="mi_dqn", parameters={"z": "dynamic"}),
    MethodSpec(name="augmented_max_avg_zx", family="augmented", parameters={"z": "dynamic", "mode": "max_avg"}),
    MethodSpec(name="augmented_max_zx", family="augmented", parameters={"z": "dynamic", "mode": "max"}),
    MethodSpec(name="augmented_average_zx", family="augmented", parameters={"z": "dynamic", "mode": "average"}),
    MethodSpec(name="tabular_augmented_max_avg_zx", family="tabular_augmented", parameters={"z": "dynamic", "mode": "max_avg"}),
    MethodSpec(name="tabular_augmented_max_zx", family="tabular_augmented", parameters={"z": "dynamic", "mode": "max"}),
    MethodSpec(name="tabular_augmented_average_zx", family="tabular_augmented", parameters={"z": "dynamic", "mode": "average"}),
    # Dyna-Q variants (model-based planning, tabular)
    MethodSpec(name="dyna_reldec", family="dyna", parameters={"reward": "mean_neighbor_sign", "z": "dynamic"}),
    MethodSpec(name="dyna_reldelta", family="dyna", parameters={"reward": "reldec_delta", "z": "dynamic"}),
    MethodSpec(name="dyna_mi", family="dyna", parameters={"reward": "mi_local", "z": "dynamic"}),
    MethodSpec(name="dyna_midelta", family="dyna", parameters={"reward": "mi_delta_local", "z": "dynamic"}),
)


def supported_method_names() -> list[str]:
    return [spec.name for spec in METHOD_CATALOG]


def supported_method_specs() -> list[MethodSpec]:
    return list(METHOD_CATALOG)


def supported_training_policy_names() -> list[str]:
    return [spec.name for spec in TRAINING_POLICY_CATALOG]


def supported_training_policy_specs() -> list[MethodSpec]:
    return list(TRAINING_POLICY_CATALOG)


def training_policy_spec(name: str) -> MethodSpec:
    normalized = str(name).strip()
    for spec in TRAINING_POLICY_CATALOG:
        if spec.name == normalized:
            return spec
    available = ", ".join(supported_training_policy_names())
    raise KeyError(f"Unknown training policy '{name}'. Available policies: {available}")


def methods_requiring_q_table(methods: list[str]) -> list[str]:
    """Return methods that require a Q-table checkpoint."""
    q_table_methods = {
        "reldec",
        "reldec_misq_local",
        "reldec_misq_global",
        "rel_delta",
        "dyna_reldec",
        "dyna_reldelta",
        "dyna_mi",
        "dyna_midelta",
    }
    return [m for m in methods if m in q_table_methods]


def methods_requiring_mi_tabular_q_table(methods: list[str]) -> list[str]:
    """Return methods that require an MI tabular Q-table checkpoint."""
    mi_tabular_methods = {"mi_tabular_zx"}
    return [m for m in methods if m in mi_tabular_methods]


def methods_requiring_tabular_augmented_q_table(methods: list[str]) -> list[str]:
    """Return methods that require a tabular augmented Q-table checkpoint."""
    tabular_augmented_methods = {
        "tabular_augmented_max_avg_zx",
        "tabular_augmented_max_zx",
        "tabular_augmented_average_zx",
    }
    return [m for m in methods if m in tabular_augmented_methods]

def methods_requiring_deep_checkpoint(methods: list[str]) -> list[str]:
    """Return methods that require a deep learning checkpoint."""
    deep_methods = {
        "deep_reldec_z1",
        "deep_reldec_zx",
        "mi_dqn_zx",
        "augmented_max_avg_zx",
        "augmented_max_zx",
        "augmented_average_zx",
    }
    return [m for m in methods if m in deep_methods]


def methods_by_family(methods: list[str]) -> dict[str, list[str]]:
    """Group methods by family."""
    families: dict[str, list[str]] = {}
    for method in methods:
        for spec in METHOD_CATALOG:
            if spec.name == method:
                if spec.family not in families:
                    families[spec.family] = []
                families[spec.family].append(method)
                break
    return families


def method_spec(name: str) -> MethodSpec:
    normalized = str(name).strip()
    for spec in METHOD_CATALOG:
        if spec.name == normalized:
            return spec
    available = ", ".join(supported_method_names())
    raise KeyError(f"Unknown method '{name}'. Available methods: {available}")