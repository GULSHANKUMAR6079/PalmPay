from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite for the prototype -- swap DATABASE_URL for Postgres before any
# real pilot. Encrypt the DB at rest either way; it holds palm embeddings
# and payment identifiers.
DATABASE_URL = "sqlite:///./palmpay.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
