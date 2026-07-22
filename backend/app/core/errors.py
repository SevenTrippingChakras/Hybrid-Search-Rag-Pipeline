"""Centralized error handling: one canonical response envelope for every error.

Domain code raises an ``AppError`` subclass; the registered handlers turn it -
along with validation errors, framework HTTP errors, and unhandled crashes - into
the single shape ``{"error": {"code", "message", "details"?}}``. Routes therefore
never need ``try/except``; they just call the service and let it raise.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("hybrid_rag")


class AppError(Exception):
    """Base domain error. Subclasses set the status, code, and default message."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "Internal server error"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.message
        super().__init__(self.message)


class DocumentNotFound(AppError):
    status_code = 404
    code = "document_not_found"
    message = "Document not found"


class UploadNotFound(AppError):
    status_code = 409
    code = "upload_not_found"
    message = "No uploaded object found for this document"


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
) -> JSONResponse:
    """Render the single, canonical error envelope ``{"error": {code, message}}``."""
    error: dict = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


def _field_errors(exc: RequestValidationError) -> list[dict]:
    """Flatten Pydantic errors to ``[{"field", "message"}]``.

    ``loc`` looks like ``("body", "size")``; drop the source prefix so the client
    sees the field path it sent.
    """
    details = []
    for err in exc.errors():
        loc = err["loc"]
        parts = loc[1:] if loc and loc[0] in ("body", "query", "path") else loc
        field = ".".join(str(p) for p in parts) or "__root__"
        details.append({"field": field, "message": err["msg"]})
    return details


def register_error_handlers(app: FastAPI) -> None:
    """Install handlers that render every error as the standard envelope."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            422,
            "validation_error",
            "Request validation failed",
            details=_field_errors(exc),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # Framework-raised HTTP errors (unknown routes, missing credentials, etc.).
        return error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", "Internal server error")
