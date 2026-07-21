"""Round-trip check for the object storage adapter against the real bucket.

Puts a small object, reads it back, verifies the bytes, then deletes it, so a
run confirms the S3 credentials and connectivity end to end:

    uv run python scripts/check_storage.py
"""

from hybrid_rag.storage import build_storage


def main() -> None:
    storage = build_storage()
    key = "_healthcheck/roundtrip.txt"
    payload = b"hybrid-rag storage round-trip"

    storage.put(key, payload, content_type="text/plain")
    assert storage.exists(key), "object missing after put"
    assert storage.get(key) == payload, "bytes did not match after get"
    storage.delete(key)
    assert not storage.exists(key), "object still present after delete"

    print("storage round-trip OK: put -> get -> delete verified")


if __name__ == "__main__":
    main()
