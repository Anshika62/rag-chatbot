import os
import logging

import pdfplumber

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.embedding import EmbeddingManager
from app.utils.vector_store import Vectorstore

from app.repository.document_repo import (
    create_document,
    create_chunks
)

from app.repository.conversation_repo import (
    get_conversation,
    get_last_10_messages,
    create_message,
    create_conversation
)

from app.service.external.LLM_service import (
    generate_answer,
    generate_answer_stream,
    generate_title
)


logger = logging.getLogger(__name__)


embedding_manager = EmbeddingManager()


# Document vector store
vector_store = Vectorstore(
    collection_name="pdf_documents"
)


# Conversation history vector store
conversation_vector_store = Vectorstore(
    collection_name="conversation_history"
)


UPLOAD_DIR = "Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_document(
    file,
    db,
    conversation_id: int,
    user_id: int
):
    try:

        # 1. Validate file
        if not file or not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is required"
            )

        # 2. Save uploaded file
        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # 3. Extract PDF text
        text = ""

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:
                text += page.extract_text() or ""

        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the file"
            )

        # 4. Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        chunks = text_splitter.split_text(text)

        # 5. Generate embeddings
        embeddings = embedding_manager.generate_embedding(
            chunks
        )

        # 6. Save chunks in vector store
        vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            filename=file.filename,
            conversation_id=conversation_id
        )

        # 7. Save document in database
        document = create_document(
            db=db,
            file_name=file.filename,
            user_id=user_id,
            conversation_id=conversation_id
        )

        # 8. Save chunks in database
        create_chunks(
            db=db,
            doc_id=document.id,
            chunks=chunks
        )

        return {
            "message": "Document processed successfully",
            "document_id": document.id,
            "conversation_id": conversation_id,
            "filename": file.filename,
            "chunks": len(chunks)
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Document processing failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process document"
        )


def query_documents(
    question: str,
    db: Session,
    user_id: int,
    conversation_id: int
):
    try:

        # 1. Validate question
        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        # 2. Validate conversation
        conversation = get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id
        )

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        # 3. Get chat history
        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )

        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in previous_messages
        ]

        # 4. Generate query embedding
        query_embedding = embedding_manager.generate_embedding(
            [question]
        )

        # 5. Search relevant documents
        results = vector_store.search(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=3
        )

        # 6. Extract contexts
        contexts = [
            point.payload["text"]
            for point in results
        ]

        context = "\n\n".join(contexts)

        logger.info(
            "Conversation query: conversation_id=%s, contexts=%s",
            conversation_id,
            len(contexts)
        )

        # 7. Generate answer
        answer = generate_answer(
            question=question,
            context=context,
            chat_history=chat_history,
            db=db,
            conversation_id=conversation_id
        )

        # 8. Save user message
        create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question
        )

        # 9. Save assistant message
        create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=answer
        )

        return {
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer,
            "contexts": contexts
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "RAG query failed: conversation_id=%s, user_id=%s",
            conversation_id,
            user_id
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def query_documents_stream(
    question: str,
    db: Session,
    user_id: int,
    conversation_id: int
):
    """
    Streaming RAG conversation flow.

    SSE events:
        start
        delta
        done
        error
    """

    try:

        # --------------------------------------------------
        # 1. Validate question
        # --------------------------------------------------

        if not question or not question.strip():

            yield {
                "event": "error",
                "success": False,
                "error_code": "BAD_REQUEST",
                "conversation_id": conversation_id,
                "message_id": None,
                "delta": None,
                "text_content": "Question cannot be empty"
            }

            return

        # --------------------------------------------------
        # 2. Validate conversation
        # --------------------------------------------------

        conversation = get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id
        )

        if not conversation:

            yield {
                "event": "error",
                "success": False,
                "error_code": "NOT_FOUND",
                "conversation_id": conversation_id,
                "message_id": None,
                "delta": None,
                "text_content": "Conversation not found"
            }

            return

        # --------------------------------------------------
        # 3. Get conversation history
        # --------------------------------------------------

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )

        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in previous_messages
        ]

        # --------------------------------------------------
        # 4. Generate embedding
        # --------------------------------------------------

        query_embedding = embedding_manager.generate_embedding(
            [question]
        )

        # --------------------------------------------------
        # 5. Search relevant documents
        # --------------------------------------------------

        results = vector_store.search(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=3
        )

        # --------------------------------------------------
        # 6. Extract contexts
        # --------------------------------------------------

        contexts = [
            point.payload["text"]
            for point in results
        ]

        context = "\n\n".join(contexts)

        logger.info(
            "Starting conversation stream: "
            "conversation_id=%s, user_id=%s, contexts=%s",
            conversation_id,
            user_id,
            len(contexts)
        )

        # --------------------------------------------------
        # 7. Send start event
        # --------------------------------------------------

        yield {
            "event": "start",
            "success": True,
            "error_code": None,
            "conversation_id": conversation_id,
            "message_id": None,
            "delta": None,
            "text_content": ""
        }

        # --------------------------------------------------
        # 8. Save user message
        # --------------------------------------------------

        user_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question
        )

        # --------------------------------------------------
        # 9. Stream LLM response
        # --------------------------------------------------

        full_answer = ""

        for chunk in generate_answer_stream(
            question=question,
            context=context,
            chat_history=chat_history
        ):

            if not chunk:
                continue

            full_answer += chunk

            yield {
                "event": "delta",
                "success": True,
                "error_code": None,
                "conversation_id": conversation_id,
                "message_id": user_message.id,
                "delta": chunk,
                "text_content": full_answer
            }

        # --------------------------------------------------
        # 10. Save assistant response
        # --------------------------------------------------

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer
        )

        # --------------------------------------------------
        # 11. Send done event
        # --------------------------------------------------

        yield {
            "event": "done",
            "success": True,
            "error_code": None,
            "conversation_id": conversation_id,
            "message_id": assistant_message.id,
            "delta": None,
            "text_content": full_answer
        }

        logger.info(
            "Conversation stream completed: conversation_id=%s",
            conversation_id
        )

    except Exception as e:

        # IMPORTANT:
        # This prints the complete traceback in Render logs.
        logger.exception(
            "QUERY STREAM ERROR: "
            "conversation_id=%s, user_id=%s",
            conversation_id,
            user_id
        )

        yield {
            "event": "error",
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "conversation_id": conversation_id,
            "message_id": None,
            "delta": None,
            "text_content": "Internal server error"
        }


def handle_conversation(
    question: str,
    db: Session,
    user_id: int,
    conversation_id: int | None
):
    try:

        # --------------------------------------------------
        # 1. Create / get conversation
        # --------------------------------------------------

        if conversation_id is None:

            title = generate_title(question)

            conversation = create_conversation(
                db=db,
                user_id=user_id,
                title=title
            )

            conversation_id = conversation.id

        else:

            conversation = get_conversation(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id
            )

            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )

        # --------------------------------------------------
        # 2. Get chat history
        # --------------------------------------------------

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )

        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in previous_messages
        ]

        # --------------------------------------------------
        # 3. Generate embedding
        # --------------------------------------------------

        query_embedding = embedding_manager.generate_embedding(
            [question]
        )

        # --------------------------------------------------
        # 4. Search documents
        # --------------------------------------------------

        results = vector_store.search(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=3
        )

        # --------------------------------------------------
        # 5. Extract contexts
        # --------------------------------------------------

        contexts = [
            point.payload["text"]
            for point in results
        ]

        context = "\n\n".join(contexts)

        # --------------------------------------------------
        # 6. Generate answer
        # --------------------------------------------------

        answer = generate_answer(
            question=question,
            context=context,
            chat_history=chat_history,
            db=db,
            conversation_id=conversation_id
        )

        # --------------------------------------------------
        # 7. Save messages
        # --------------------------------------------------

        create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question
        )

        create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=answer
        )

        return {
            "conversation_id": conversation_id,
            "title": conversation.title,
            "question": question,
            "answer": answer,
            "contexts": contexts
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "Conversation handling failed: "
            "conversation_id=%s, user_id=%s",
            conversation_id,
            user_id
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )