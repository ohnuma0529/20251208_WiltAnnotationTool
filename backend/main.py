from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .config import settings
from .api.endpoints import router as api_router
from .core.model_loader import model_loader
import os

app = FastAPI(title="Tomato Wilt Annotation Tool")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router, prefix="/api")

# Static files for images (to serve them to frontend)
# We mount the image directory directly
app.mount("/images", StaticFiles(directory=settings.CACHE_DIR), name="images")

@app.on_event("startup")
async def startup_event():
    # Load models on startup
    model_loader.load_models() 
    # Commented out to prevent crash during development if weights missing.
    # User should enable this.
    print("Application starting...")

@app.get("/")
def read_root():
    return {"message": "Tomato Wilt Annotation Tool API is ready"}
