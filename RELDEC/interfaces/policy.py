from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    @abstractmethod
    def act(self, state: Any, *, deterministic: bool = False) -> Any:
        raise NotImplementedError

    @abstractmethod
    def serialize_config(self) -> dict[str, Any]:
        raise NotImplementedError