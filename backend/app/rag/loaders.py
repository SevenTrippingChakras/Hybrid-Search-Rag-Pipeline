"""Multi-format document loaders.

Normalize markdown, plain text, HTML, and PDF files into clean-plaintext
`Segment`s carrying structural metadata (source, heading, page). Raw uploads
are copied to ``data/raw`` and the normalized segments are cached to
``data/processed`` so the corpus can be re-indexed without re-parsing.
"""

import json
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.rag.models import Segment

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_HTML_TEXT_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"]


def load_file(path: str | Path) -> list[Segment]:
    """Parse a file into normalized segments, dispatching on extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix!r}")
    return loader(path)


def is_supported(filename: str) -> bool:
    """True if a file with this name can be parsed by the loaders."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def load_bytes(data: bytes, filename: str) -> list[Segment]:
    """Parse in-memory file bytes by writing them to a temp file named ``filename``.

    Uploaded documents live in object storage as bytes, not on disk. Writing them
    to a temp file that keeps the original name lets the path-based loaders run
    unchanged and carry the right ``source``. Any path parts in ``filename`` are
    stripped for safety.
    """
    name = Path(filename).name
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        path.write_bytes(data)
        return load_file(path)


def ingest_file(path: str | Path, data_dir: str | Path = "data") -> list[Segment]:
    """Load a file, archive the raw copy, and cache normalized segments.

    Returns the parsed segments. Re-running against the cached processed JSON
    (see :func:`load_processed`) avoids re-parsing on re-index.
    """
    path = Path(path)
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_copy = raw_dir / path.name
    if path.resolve() != raw_copy.resolve():
        shutil.copy2(path, raw_copy)

    segments = load_file(path)
    processed_path = processed_dir / f"{path.stem}.json"
    processed_path.write_text(
        json.dumps([asdict(s) for s in segments], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return segments


def load_processed(processed_path: str | Path) -> list[Segment]:
    """Rehydrate segments from a cached processed JSON file."""
    data = json.loads(Path(processed_path).read_text(encoding="utf-8"))
    return [Segment(**item) for item in data]


def _load_text(path: Path) -> list[Segment]:
    text = _normalize(path.read_text(encoding="utf-8"))
    return [Segment(text=text, source=path.name)] if text else []


def _load_pdf(path: Path) -> list[Segment]:
    reader = PdfReader(str(path))
    segments = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _normalize(page.extract_text() or "")
        if text:
            segments.append(Segment(text=text, source=path.name, page=page_number))
    return segments


def _load_markdown(path: Path) -> list[Segment]:
    segments: list[Segment] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = _normalize("\n".join(buffer))
        if text:
            segments.append(Segment(text=text, source=path.name, heading=heading))
        buffer.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return segments


def _load_html(path: Path) -> list[Segment]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    segments: list[Segment] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = _normalize("\n".join(buffer))
        if text:
            segments.append(Segment(text=text, source=path.name, heading=heading))
        buffer.clear()

    root = soup.body or soup
    for element in root.find_all(_HTML_TEXT_TAGS):
        if element.name.startswith("h"):
            flush()
            heading = element.get_text(" ", strip=True)
        else:
            text = element.get_text(" ", strip=True)
            if text:
                buffer.append(text)
    flush()
    return segments


_LOADERS = {
    ".pdf": _load_pdf,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".html": _load_html,
    ".htm": _load_html,
    ".txt": _load_text,
    ".text": _load_text,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)


def _normalize(text: str) -> str:
    """Strip trailing whitespace and collapse runs of blank lines."""
    out: list[str] = []
    blank = False
    for line in text.splitlines():
        line = line.rstrip()
        if line:
            out.append(line)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()
