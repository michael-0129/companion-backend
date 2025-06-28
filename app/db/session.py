from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings # Import settings from the new config file

# The database URL is now sourced from settings
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# Create SQLAlchemy engine
# For asyncpg, connect_args might not be needed unless specific SSL or other params are required.
# If your DB is indeed PostgreSQL and you intend to use async operations with SQLAlchemy 1.4+ / 2.0,
# you should use an async engine. However, the current get_db and service patterns appear synchronous.
# For now, assuming synchronous engine setup as before, but with asyncpg dialect.
# If full async is intended, this needs to be an AsyncEngine and SessionLocal needs to be async_sessionmaker.

if "asyncpg" in SQLALCHEMY_DATABASE_URL:
    # Standard practice for SQLAlchemy 2.0 style with future=True for 1.4
    engine = create_engine(SQLALCHEMY_DATABASE_URL, future=True) 
    # For actual async operations, you'd use: from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    # engine = create_async_engine(SQLALCHEMY_DATABASE_URL)
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a SessionLocal class
# This will be used to create database sessions.
# Again, if fully async: from sqlalchemy.ext.asyncio import async_sessionmaker
# SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class
# All database models will inherit from this class.
Base = declarative_base()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to create database tables (optional, can be handled by Alembic)
# def init_db():
#     Base.metadata.create_all(bind=engine) 