# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from photon.api.app import create_app
from photon.config import PhotonConfig
from tests.test_config import VALID


@pytest.fixture
def config() -> PhotonConfig:
    return PhotonConfig.model_validate(VALID)


@pytest.fixture
def client(config, tmp_path):
    app = create_app(
        config=config,
        db_path=str(tmp_path / "photon.db"),
        registry_db=str(tmp_path / "registry.db"),
        shadow_db=str(tmp_path / "shadow.db"),
        tokeff_db=str(tmp_path / "tokeff.db"),
    )
    # `with` runs the lifespan (creates/closes the shared httpx client)
    with TestClient(app) as c:
        yield c
