"""Live round-trip for presigned PUT uploads against the real bucket.

Signs a PUT URL, uploads bytes straight to that URL with an HTTP client holding
no S3 credentials (exactly how the browser will), then confirms the object
landed and cleans up. Proves the presigned upload path end to end:

    uv run python scripts/check_presigned.py
"""

import httpx

from app.storage import build_storage


def main() -> None:
    storage = build_storage()
    key = "_healthcheck/presigned.txt"
    payload = b"hybrid-rag presigned round-trip"
    content_type = "text/plain"

    url = storage.presigned_put_url(key, content_type=content_type)

    # No auth here: the signature in the URL is the only credential, and the
    # Content-Type header must match what was signed in.
    resp = httpx.put(url, content=payload, headers={"Content-Type": content_type})
    resp.raise_for_status()

    assert storage.exists(key), "object missing after presigned put"
    assert storage.get(key) == payload, "bytes did not match after presigned put"
    storage.delete(key)

    print("presigned round-trip OK: sign -> client PUT -> verify -> delete")


if __name__ == "__main__":
    main()
