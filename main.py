"""
FastAPI Server Entrypoint with modern lifespan handler and CORS configuration
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.model import model_manager
from app.routers.assets import router as assets_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Execute model load/training on server startup
    model_manager.load_or_train()
    yield


app = FastAPI(
    title="ITAD-Vision: Enterprise Asset Depreciation API",
    description="REST API providing predictive valuation and NIST 800-88 ITAD governance workflows.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assets_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)