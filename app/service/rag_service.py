import os
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
    create_message
)

from app.service.external.LLM_service import generate_answer, generate_answer_stream, generate_title
from app.repository.conversation_repo import create_conversation


embedding_manager = EmbeddingManager()


# PDF / document vector store
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
    user_id:int
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

        # 6. Save document chunks in Qdrant
        #    with conversation_id
        vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            filename=file.filename,
            conversation_id=conversation_id
        )

        # 7. Save document in Neon DB
        document = create_document(
            db=db,
            file_name=file.filename,
            user_id=user_id,
            conversation_id=conversation_id
        )

        # 8. Save chunks in Neon DB
        create_chunks(
            db=db,
            doc_id=document.id,
            chunks=chunks
        )

        # 9. Return response
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

        if not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

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


        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )


        chat_history = []

        for message in previous_messages:
            chat_history.append({
                "role": message.role,
                "content": message.content
            })

        query_embedding = embedding_manager.generate_embedding(
            [question]
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=3
        )

        contexts = []

        for point in results:
            contexts.append(
                point.payload["text"]
            )

        context = "\n\n".join(contexts)

        print("CHAT HISTORY:")
        print(chat_history)

        answer = generate_answer(
              question=question,
              context=context,
              chat_history=chat_history,
              db=db,
              conversation_id=conversation_id
        )

        user_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question
        )

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=answer
        )

        # 11. Return response
        return {
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer,
            "contexts": contexts
        }

    except HTTPException:
        raise

    except Exception as e:
        print("RAG SERVICE ERROR:", repr(e))
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
    try:
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

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )

        chat_history = [
            {
                "role": m.role,
                "content": m.content
            }
            for m in previous_messages
        ]

        query_embedding = embedding_manager.generate_embedding(
            [question]
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=3
        )

        contexts = [
            point.payload["text"]
            for point in results
        ]

        context = "\n\n".join(contexts)

        create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question
        )

        yield {
            "event": "start",
            "success": True,
            "error_code": None,
            "conversation_id": conversation_id,
            "message_id": None,
            "delta": None,
            "text_content": ""
        }

        full_answer = ""

        for chunk in generate_answer_stream(
            question=question,
            context=context,
            chat_history=chat_history
        ):
            full_answer += chunk

            yield {
                "event": "delta",
                "success": True,
                "error_code": None,
                "conversation_id": conversation_id,
                "message_id": None,
                "delta": chunk,
                "text_content": full_answer
            }

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer
        )

        yield {
            "event": "done",
            "success": True,
            "error_code": None,
            "conversation_id": conversation_id,
            "message_id": assistant_message.id,
            "delta": None,
            "text_content": full_answer
        }

    except Exception as e:
        print("QUERY STREAM ERROR:", repr(e))

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

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )

        chat_history = [
            {
                "role": m.role,
                "content": m.content
            }
            for m in previous_messages
        ]

        query_embedding = embedding_manager.generate_embedding(
            [question]
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=3
        )

        contexts = [
            point.payload["text"]
            for point in results
        ]

        context = "\n\n".join(contexts)

        answer = generate_answer(
             question=question,
             context=context,
             chat_history=chat_history,
             db=db,
             conversation_id=conversation_id
        )

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )