from pathlib import Path

from backend.app.models import ModelTier
from backend.app.routing import ModelCatalog


def test_default_model_catalog_has_each_conceptual_tier() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    catalog = ModelCatalog.from_yaml(path)

    assert {model.tier for model in catalog.models} == set(ModelTier)
    assert len(catalog.enabled_models) == 4
