from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.services.vector_store import vector_store
from app.storage.db import initialize_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    database = initialize_database()
    database.close()
    vector_store.initialize()
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.include_router(router)
