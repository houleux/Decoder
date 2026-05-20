from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ActionSpace(ABC):
    @abstractmethod
    def valid_actions(self, observation: dict[str, Any]) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def mask(self, observation: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, action_id: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def serialize_config(self) -> dict[str, Any]:
        raise NotImplementedError