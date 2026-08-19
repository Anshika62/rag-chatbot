from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.document import Document
from app.models.document_chunks import Docs_chunks


def create_document(
    db: Session,
    file_name: str,
    conversation_id: int
):
    try:
        document = Document(
            file_name=file_name,
            conversation_id=conversation_id
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return document

    except SQLAlchemyError:
        db.rollback()
        raise


def create_chunks(
    db: Session,
    doc_id: int,
    chunks: list[str]
):
    try:
        chunk_objects = [
            Docs_chunks(
                doc_id=doc_id,
                chunk_text=chunk
            )
            for chunk in chunks
        ]

        db.add_all(chunk_objects)
        db.commit()

        return chunk_objects

    except SQLAlchemyError:
        db.rollback()
        raise


def get_document_by_id(
    db: Session,
    doc_id: int
):
    try:
        return (
            db.query(Document)
            .filter(
                Document.id == doc_id
            )
            .first()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


def get_chunks_by_document_id(
    db: Session,
    doc_id: int
):
    try:
        return (
            db.query(Docs_chunks)
            .filter(
                Docs_chunks.doc_id == doc_id
            )
            .all()
        )

    except SQLAlchemyError:
        db.rollback()
        raise