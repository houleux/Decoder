from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    @abstractmethod
    def train(self, run_config: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError