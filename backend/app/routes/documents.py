"""Document upload endpoints (presigned flow).

Thin: parse the request, call the service, shape the response, translate domain
errors to HTTP. No business logic here. Request/response schemas live with the
route; the persisted shape is ``models.document.Document``.
"""

from fastapi import APIRouter, HTTPException, status

from app.deps import DocumentServiceDep
from app.models.document import (
    Document,
    InitiateUploadRequest,
    InitiateUploadResponse,
)
from app.services.document_service import DocumentNotFound, UploadNotFound

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def initiate_upload(
    body: InitiateUploadRequest, service: DocumentServiceDep
) -> InitiateUploadResponse:
    """Create a pending record and return a presigned URL to PUT the file to."""
    document, upload_url = await service.initiate_upload(
        filename=body.filename,
        content_type=body.content_type,
        size=body.size,
    )
    return InitiateUploadResponse(
        document_id=document.document_id, upload_url=upload_url
    )


@router.post("/{document_id}/complete")
async def complete_upload(document_id: str, service: DocumentServiceDep) -> Document:
    """Confirm the upload landed in storage and mark the document uploaded."""
    try:
        return await service.complete_upload(document_id)
    except DocumentNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found") from None
    except UploadNotFound:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "no uploaded object found for this document"
        ) from None


@router.get("")
async def list_documents(service: DocumentServiceDep) -> list[Document]:
    return await service.list_documents()


@router.get("/{document_id}")
async def get_document(document_id: str, service: DocumentServiceDep) -> Document:
    try:
        return await service.get_document(document_id)
    except DocumentNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found") from None


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, service: DocumentServiceDep) -> None:
    try:
        await service.delete_document(document_id)
    except DocumentNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found") from None
