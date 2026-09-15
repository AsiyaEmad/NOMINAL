from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from backend.app.models import ModelDefinition


class ModelCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: list[ModelDefinition] = Field(default_factory=list)

    @property
    def enabled_models(self) -> list[ModelDefinition]:
        return [model for model in self.models if model.enabled]

    def get(self, model_id_or_name: str) -> ModelDefinition | None:
        return next(
            (
                model
                for model in self.models
                if model.id == model_id_or_name or model.model_name == model_id_or_name
            ),
            None,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelCatalog":
        with path.open(encoding="utf-8") as catalog_file:
            document = yaml.safe_load(catalog_file) or {}
        return cls.model_validate(document)
