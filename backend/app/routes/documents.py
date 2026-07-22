"""Document upload endpoints (presigned flow).

Thin: parse the request, call the service, shape the response. Domain errors the
service raises (``AppError`` subclasses) are turned into the standard error
envelope by the handlers in ``core.errors`` - so no ``try/except`` here.
"""

from fastapi import APIRouter, status

from app.deps import DocumentServiceDep
from app.models.document import (
    Document,
    InitiateUploadRequest,
    InitiateUploadResponse,
)

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
    return await service.complete_upload(document_id)


@router.get("")
async def list_documents(service: DocumentServiceDep) -> list[Document]:
    return await service.list_documents()


@router.get("/{document_id}")
async def get_document(document_id: str, service: DocumentServiceDep) -> Document:
    return await service.get_document(document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, service: DocumentServiceDep) -> None:
    await service.delete_document(document_id)
