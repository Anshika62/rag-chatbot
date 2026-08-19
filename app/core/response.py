from typing import Any
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def success_response(
    message: str,
    data: Any = None,
    status_code: int = 200
):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": True,
            "status_code": status_code,
            "message": message,
            "data": data,
            "error_code": None
        }
    )

class AppException(HTTPException):

    def __init__(self, status_code: int, message: str, error_code: str):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code

def error_response(
    message: str,
    error_code: str,
    status_code: int,
    data: Any = None
):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "status_code": status_code,
            "message": message,
            "data": data,
            "error_code": error_code
        }
    )



def stream_response(
    event: str,
    message: str,
    data: Any = None,
    status_code: int = 200,
    error_code: str | None = None
):
    return {
        "event": event,
        "success": error_code is None,
        "status_code": status_code,
        "message": message,
        "data": data,
        "error_code": error_code
    }



async def http_exception_handler(
    request: Request,
    exc: HTTPException
):
    error_codes = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT",
    }

    error_code = getattr(
        exc,
        "error_code",
        error_codes.get(exc.status_code, "HTTP_ERROR")
    )

    return error_response(
        message=str(exc.detail),
        error_code=error_code,
        status_code=exc.status_code
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    return error_response(
        message="Request validation failed",
        error_code="VALIDATION_ERROR",
        status_code=422,
        data={
            "errors": jsonable_encoder(exc.errors())
        }
    )


async def global_exception_handler(
    request: Request,
    exc: Exception
):
    print("UNHANDLED ERROR:", repr(exc))

    return error_response(
        message="Internal server error",
        error_code="INTERNAL_SERVER_ERROR",
        status_code=500
    )



def setup_exception_handlers(app: FastAPI): 

    app.add_exception_handler(
        HTTPException,
        http_exception_handler
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler
    )

    app.add_exception_handler(
        Exception,
        global_exception_handler
    )