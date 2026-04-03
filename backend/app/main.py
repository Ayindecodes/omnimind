import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.database import Base, engine
from app.logging_config import setup_logging
from app.api.routes import chat, health, memory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    if settings.bootstrap_schema:
        Base.metadata.create_all(bind=engine)
        logger.warning(
            "bootstrap_schema=true: created missing tables (use Alembic for production)"
        )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(memory.router, prefix="/api/v1")
    return app


app = create_app()
