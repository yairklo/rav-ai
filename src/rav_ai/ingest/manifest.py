"""Checksum manifest of an export snapshot (kept outside git, next to the raw data)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rav_ai.ingest.sefaria import export_label, file_sha256


def write_manifest(raw_dir: Path, out_dir: Path, log=print) -> Path:
    files = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and not p.name.endswith(".part") and "merged.json" != p.name:
            files.append(
                {
                    "path": p.relative_to(raw_dir).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
            )
    label = export_label(raw_dir)
    doc = {"export_label": label, "created": datetime.now(UTC).isoformat(), "files": files}
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    out = out_dir / f"sefaria-{safe}.manifest.json"
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log(f"manifest: {len(files)} files -> {out}")
    return out
