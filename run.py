import uvicorn
from app.core.config import settings # To potentially use settings for host/port if defined

if __name__ == "__main__":
    # It's good practice to load .env before app import if settings aren't fully initialized
    # However, Pydantic settings should handle .env loading upon instantiation.
    # from dotenv import load_dotenv
    # load_dotenv() # Ensure .env is loaded if not already by settings

    # The app should be imported after any necessary environment setup
    from app.main import app

    # You can make host, port, log_level configurable here, perhaps via settings
    # For now, using common defaults.
    host = "0.0.0.0" # Listen on all available network interfaces
    port = 8000
    reload = True # Enable auto-reload for development

    print(f"Starting Uvicorn server on http://{host}:{port}")
    uvicorn.run(
        "app.main:app", # Path to the FastAPI app instance
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    ) 