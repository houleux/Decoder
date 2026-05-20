from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, run_config: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def summarize(self) -> dict[str, Any]:
        raise NotImplementedError