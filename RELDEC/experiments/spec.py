from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    code: str
    matrix_csv: str
    policy_type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "matrix_csv": self.matrix_csv,
            "policy_type": self.policy_type,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    experiment: ExperimentSpec
    training_config: dict[str, Any]
    artifacts: dict[str, Any]

    @staticmethod
    def create(
        run_id: str,
        experiment: ExperimentSpec,
        training_config: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> "RunManifest":
        return RunManifest(
            run_id=str(run_id),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            experiment=experiment,
            training_config=dict(training_config),
            artifacts=dict(artifacts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "experiment": self.experiment.to_dict(),
            "training_config": dict(self.training_config),
            "artifacts": dict(self.artifacts),
        }


@dataclass(frozen=True)
class EvaluationSpec:
    code: str
    matrix_csv: str
    methods: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "matrix_csv": self.matrix_csv,
            "methods": list(self.methods),
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class EvaluationManifest:
    run_id: str
    created_at_utc: str
    experiment: EvaluationSpec
    evaluation_config: dict[str, Any]
    artifacts: dict[str, Any]

    @staticmethod
    def create(
        run_id: str,
        experiment: EvaluationSpec,
        evaluation_config: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> "EvaluationManifest":
        return EvaluationManifest(
            run_id=str(run_id),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            experiment=experiment,
            evaluation_config=dict(evaluation_config),
            artifacts=dict(artifacts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "experiment": self.experiment.to_dict(),
            "evaluation_config": dict(self.evaluation_config),
            "artifacts": dict(self.artifacts),
        }