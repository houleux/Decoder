from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RewardFn(ABC):
    @abstractmethod
    def compute(self, before: dict[str, Any], after: dict[str, Any], info: dict[str, Any]) -> float:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def serialize_config(self) -> dict[str, Any]:
        raise NotImplementedError