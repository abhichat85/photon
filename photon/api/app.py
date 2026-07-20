# photon/api/app.py
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from photon.api.admin import admin_router
from photon.api.chat import chat_router
from photon.api.metrics import metrics_router
from photon.backends.openai_proxy import OpenAIProxy
from photon.config import PhotonConfig
from photon.api.pipelines import pipelines_router
from photon.core.fleet import FleetManager
from photon.core.serving import VLLMServingBackend
from photon.core.shadow_store import ShadowDecisionStore
from photon.observability import (
    RequestLogMiddleware,
    init_sentry,
    init_tracing,
    setup_logging,
)
from photon.registry.store import RegistryStore
from photon.router.shadow import ShadowPolicy
from photon.router.static import StaticRouter
from photon.telemetry.store import TelemetryStore


def create_app(
    config: PhotonConfig | None = None,
    db_path: str | None = None,
    registry_db: str | None = None,
    shadow_db: str | None = None,
) -> FastAPI:
    config = config or PhotonConfig.from_yaml(os.environ["PHOTON_CONFIG"])
    db_path = db_path or os.environ.get("PHOTON_DB", "photon.db")
    registry_db = registry_db or os.environ.get("PHOTON_REGISTRY_DB", "registry.db")
    shadow_db = shadow_db or os.environ.get("PHOTON_SHADOW_DB", "shadow.db")

    setup_logging()
    init_sentry()  # no-op unless SENTRY_DSN is set

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
        app.state.proxy = OpenAIProxy(app.state.http)
        # pipeline execution runs against the ServingBackend seam; the Tier-2
        # dense engine replaces this construction without touching the API
        app.state.serving_backend = VLLMServingBackend(
            {b.name: b for b in config.backends}, app.state.http
        )
        yield
        await app.state.http.aclose()

    app = FastAPI(title="photon", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestLogMiddleware)
    init_tracing(app)  # no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set
    app.state.config = config
    app.state.router = StaticRouter(config)
    app.state.shadow = ShadowPolicy(config)
    app.state.store = TelemetryStore(db_path)
    app.state.registry = RegistryStore(registry_db)
    app.state.fleet_manager = FleetManager()
    app.state.fleet_plan = None  # set by POST /photon/v1/fleet
    # Shadow study plumbing: the durable store always exists (cheap); the shadow
    # router itself stays opt-in. Enable with:
    #   app.state.shadow_router = ShadowRouter(learned, sink=app.state.shadow_store.insert)
    app.state.shadow_store = ShadowDecisionStore(shadow_db)
    app.state.shadow_router = None  # set to a ShadowRouter to enable shadow logging
    app.state.pipelines = {}  # id -> PipelineSpec; per-process, config-like
    app.include_router(chat_router)
    app.include_router(admin_router)
    app.include_router(metrics_router)
    app.include_router(pipelines_router)
    return app


def main_app() -> FastAPI:
    """uvicorn entrypoint: uvicorn photon.api.app:main_app --factory"""
    return create_app()
