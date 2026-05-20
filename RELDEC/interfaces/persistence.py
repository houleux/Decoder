from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PersistenceStore(ABC):
    @abstractmethod
    def save_manifest(self, run_id: str, manifest: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, run_id: str, checkpoint: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, run_id: str) -> dict[str, Any]:
        raise NotImplementedError