from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from orchestrator.api.routes import flywheel, experiments, models, health
from orchestrator.core.config import settings
from orchestrator.core.database import connect_db, disconnect_db
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting_up", env=settings.ENV)
    await connect_db()
    yield
    await disconnect_db()
    logger.info("shut_down")


app = FastAPI(
    title="Data Flywheel Orchestrator",
    description="Continuous model distillation and improvement loop",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(flywheel.router, prefix="/flywheel", tags=["flywheel"])
app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
app.include_router(models.router, prefix="/models", tags=["models"])
