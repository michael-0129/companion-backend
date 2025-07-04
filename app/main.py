import os
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from starlette.exceptions import HTTPException as StarletteHTTPException

# Corrected imports for database objects
from app.db.session import Base, engine # Import Base and engine from the new session.py
from app.core.config import settings # Import settings
from fastapi.middleware.cors import CORSMiddleware

# Import API routers - these paths will be updated later in the refactoring
from app.api import codex, chat, protocol, documents

# Load environment variables - Pydantic settings now handles this, but load_dotenv() can remain if other non-Pydantic env vars are used elsewhere or for local dev convenience.
load_dotenv() 

# Create database tables
# This is an idempotent operation, so it's safe to run on every startup.
# For more complex migration needs, Alembic would be the next step.
Base.metadata.create_all(bind=engine)

origins = [
    "*"  # Allow all origins for development. Change to specific URLs for production.
]
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Get the absolute path to the dist folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(BASE_DIR, "dist")
ASSETS_DIR = os.path.join(DIST_DIR, "assets")

# Mount static frontend assets
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="static")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}. API docs at /docs or /redoc."}

# Include API routers
# The prefix for these routers will be updated when we move to api_v1 structure
app.include_router(codex.router, prefix=f"{settings.API_V1_STR}/codex", tags=["Codex (Memory) Endpoints"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["Chat Endpoints"])
app.include_router(protocol.router, prefix=f"{settings.API_V1_STR}/protocol", tags=["Protocol Event Endpoints"])
app.include_router(documents.router, prefix=f"{settings.API_V1_STR}/documents", tags=["Documents"])

# Serve index.html at root
@app.get("/")
async def serve_root():
    return FileResponse(os.path.join(DIST_DIR, "index.html"))

# Custom 404 fallback for SPA (client-side routing)
@app.exception_handler(StarletteHTTPException)
async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404 and not any(
        request.url.path.startswith(p) for p in ["/api", "/assets"]
    ):
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "message": f"{settings.PROJECT_NAME} services are operational."} 