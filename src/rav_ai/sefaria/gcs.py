"""Minimal public-GCS access for gs://sefaria-export (no credentials needed)."""

from __future__ import annotations

import random
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import httpx

BUCKET = "sefaria-export"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}"


def backoff_sleep(attempt: int, base: float = 2.0, cap: float = 120.0) -> None:
    """Exponential backoff with jitter: ~2, 4, 8, ... capped at `cap` seconds."""
    time.sleep(min(cap, base * 2**attempt) * (0.5 + random.random() / 2))


def _client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(30.0, connect=15.0, read=60.0), follow_redirects=True)


def list_objects(prefix: str = "", client: httpx.Client | None = None) -> Iterator[dict]:
    """Yield {name, size} for every object under prefix (paginated)."""
    own = client is None
    client = client or _client()
    try:
        token = None
        while True:
            params = {
                "prefix": prefix,
                "maxResults": 1000,
                "fields": "nextPageToken,items(name,size)",
            }
            if token:
                params["pageToken"] = token
            for attempt in range(6):
                try:
                    r = client.get(API, params=params)
                    r.raise_for_status()
                    break
                except (httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == 5:
                        raise
                    backoff_sleep(attempt)
            data = r.json()
            for item in data.get("items", []):
                yield {"name": item["name"], "size": int(item["size"])}
            token = data.get("nextPageToken")
            if not token:
                return
    finally:
        if own:
            client.close()


def object_url(name: str) -> str:
    return f"{MEDIA}/{quote(name, safe='/')}"


def local_path(raw_dir: Path, name: str) -> Path:
    """Mirror the bucket layout under raw_dir. Names are kept verbatim (UTF-8)."""
    return raw_dir.joinpath(*name.split("/"))


def is_complete(dest: Path, size: int | None) -> bool:
    return dest.exists() and (size is None or dest.stat().st_size == size)


def download(
    name: str,
    raw_dir: Path,
    size: int | None = None,
    client: httpx.Client | None = None,
    retries: int = 8,
    use_gsutil: bool = False,
) -> Path:
    """Download one object, resuming from `<file>.part` via HTTP Range. Idempotent."""
    dest = local_path(raw_dir, name)
    if is_complete(dest, size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if use_gsutil:
        return _gsutil_download(name, dest, size)
    part = dest.with_name(dest.name + ".part")
    own = client is None
    client = client or _client()
    try:
        for attempt in range(retries):
            try:
                have = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={have}-"} if have else {}
                with client.stream("GET", object_url(name), headers=headers) as r:
                    if r.status_code == 416:  # already have all bytes
                        break
                    r.raise_for_status()
                    mode = "ab" if r.status_code == 206 else "wb"
                    with open(part, mode) as f:
                        for chunk in r.iter_bytes(1 << 16):
                            f.write(chunk)
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == retries - 1:
                    raise
                backoff_sleep(attempt)
        if size is not None and part.stat().st_size != size:
            part.unlink()
            raise OSError(f"size mismatch for {name}")
        part.replace(dest)
        return dest
    finally:
        if own:
            client.close()


def gsutil_available() -> bool:
    return shutil.which("gsutil") is not None or shutil.which("gcloud") is not None


def _gsutil_download(name: str, dest: Path, size: int | None) -> Path:
    """Fallback transport: `gcloud storage cp` / `gsutil cp` (both resume interrupted downloads)."""
    uri = f"gs://{BUCKET}/{name}"
    if shutil.which("gcloud"):
        cmd = ["gcloud", "storage", "cp", uri, str(dest)]
    else:
        cmd = ["gsutil", "-q", "cp", uri, str(dest)]
    subprocess.run(cmd, check=True, timeout=1800)
    if size is not None and dest.stat().st_size != size:
        raise OSError(f"size mismatch for {name}")
    return dest
