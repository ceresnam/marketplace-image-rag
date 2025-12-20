from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .const import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER, EMBEDDING_DIM


class ImageEmbedding(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    image_path: str = Field(index=True)
    embedding: list[float] = Field(sa_column=Column(Vector(EMBEDDING_DIM)))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def save_image_embeddings(image_embeddings: list[ImageEmbedding]):
    with Session(engine) as session:
        session.add_all(image_embeddings)
        session.commit()


engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    pool_size=5,
    pool_pre_ping=True,
)
SQLModel.metadata.create_all(engine)
