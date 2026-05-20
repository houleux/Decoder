from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StateEncoder(ABC):
    @abstractmethod
    def build(self, observation: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def shape(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def serialize_config(self) -> dict[str, Any]:
        raise NotImplementedError