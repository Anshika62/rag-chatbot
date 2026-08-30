import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker, declarative_base


logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env")


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # Lowered from 300s. Some managed/cloud Postgres providers drop
    # idle connections well before 300s, which was surfacing as an
    # unhandled OperationalError during db.close() on long-running
    # requests (e.g. multi-minute document/image uploads that hold
    # one session open across several slow external API calls).
    # pool_pre_ping still does the real per-checkout validation;
    # this just recycles proactively sooner as a second safety net.
    pool_recycle=180,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        # On a long-running request (e.g. streamed document/image
        # upload), the underlying DB connection can be dropped
        # server-side (idle timeout) while this session was
        # checked out. The actual work for the request has already
        # completed successfully by this point — db.close() here is
        # just cleanup. If the connection is already dead, closing
        # it can itself raise (e.g. psycopg2 "SSL connection has
        # been closed unexpectedly"), which previously surfaced as
        # an unhandled server error/traceback for an otherwise
        # successful request. Swallow that specific failure mode:
        # log it, and let SQLAlchemy's connection pool discard the
        # broken connection on its own (pool_pre_ping will validate
        # a fresh one on the next checkout).
        try:
            db.close()

        except DBAPIError:

            logger.warning(
                "DB session cleanup: underlying connection was "
                "already closed (likely idle-timeout on a "
                "long-running request). Discarding it safely."
            )