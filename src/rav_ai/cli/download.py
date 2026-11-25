"""Download (parts of) the Sefaria export into data/raw. Resumable and idempotent.

Examples:
  rav-download --book "Shulchan Arukh, Orach Chayim"
  rav-download --category Halakhah --no-links
  rav-download --all --links
  rav-download --links-only
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from rav_ai.config import RAW_DIR
from rav_ai.sefaria import gcs
from rav_ai.sefaria.layout import parse_version_object, priority_tier, schema_object_name


def select_objects(books, categories, everything, include_merged, languages, client):
    """Return ([(name, size, tier)], books): version json files + schemas, pilot sources first."""
    wanted: list[tuple[str, int, int]] = []
    book_tier: dict[str, int] = {}
    schema_books: set[str] = set()
    skipped_bad_names: list[str] = []
    for obj in gcs.list_objects("json/", client):
        try:
            obj["name"].encode("utf-8")
        except UnicodeEncodeError:
            skipped_bad_names.append(obj["name"])
            continue
        v = parse_version_object(obj["name"], obj["size"])
        if v is None or (v.is_merged and not include_merged):
            continue
        if languages and v.language_dir not in languages:
            continue
        if not (everything or v.book in books or any(c in v.category_path for c in categories)):
            continue
        tier = priority_tier(v)
        wanted.append((v.name, v.size, tier))
        book_tier[v.book] = min(tier, book_tier.get(v.book, 9))
        schema_books.add(v.book)
    for n in skipped_bad_names:
        print(f"warning: skipping object with undecodable name: {n!r}", file=sys.stderr)
    schema_sizes = {o["name"]: o["size"] for o in gcs.list_objects("schemas/", client)}
    for b in sorted(schema_books):
        n = schema_object_name(b)
        if n in schema_sizes:
            wanted.append((n, schema_sizes[n], book_tier[b]))
        else:
            print(f"warning: no schema file for {b!r}", file=sys.stderr)
    # pilot tiers first; inside a tier schemas (small, required for ingest) before texts
    wanted.sort(key=lambda t: (t[2], not t[0].startswith("schemas/")))
    return wanted, schema_books


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--book", action="append", default=[], help="exact book title (repeatable)")
    ap.add_argument(
        "--category",
        action="append",
        default=[],
        help="category path component, e.g. Halakhah, Talmud (repeatable)",
    )
    ap.add_argument("--all", action="store_true", help="every book (~3.7 GB of version files)")
    ap.add_argument(
        "--language",
        action="append",
        choices=["Hebrew", "English"],
        default=[],
        help="restrict to a language directory (default: both)",
    )
    ap.add_argument(
        "--include-merged",
        action="store_true",
        help="also fetch merged.json (no license field; not used by ingest)",
    )
    ap.add_argument("--links", action="store_true", help="also download links/*.csv (~0.7 GB)")
    ap.add_argument("--links-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--jobs", type=int, default=3, help="parallel downloads (keep low: GCS throttles)"
    )
    ap.add_argument(
        "--backend",
        choices=["auto", "http", "gsutil"],
        default="auto",
        help="auto = HTTP with exponential backoff, then gcloud/gsutil for files that still fail",
    )
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    args = ap.parse_args(argv)

    from pathlib import Path

    raw = Path(args.raw_dir)
    todo: list[tuple[str, int | None, float]] = [
        ("last_export.txt", None, -1),
        ("table_of_contents.json", None, -1),
    ]
    with gcs._client() as client:
        if not args.links_only:
            if not (args.book or args.category or args.all):
                ap.error("choose --book, --category, --all or --links-only")
            sel, _ = select_objects(
                set(args.book),
                set(args.category),
                args.all,
                args.include_merged,
                set(args.language),
                client,
            )
            todo += sel
        if args.links or args.links_only:
            # links are needed by the pilot: after pilot texts (tiers 0,1), before the rest
            todo += [
                (o["name"], o["size"], 1.5)
                for o in gcs.list_objects("links/", client)
                if o["name"].endswith(".csv")
            ]
    todo.sort(key=lambda t: t[2])  # stable: keeps schema-before-text order inside a tier

    pending = [(n, s) for n, s, _ in todo if not gcs.is_complete(gcs.local_path(raw, n), s)]
    total = sum(s or 0 for _, s in pending)
    print(f"{len(todo)} objects selected, {len(pending)} to download ({total / 1e6:.1f} MB)")
    if args.dry_run:
        for n, s in pending[:50]:
            print(" ", n, s)
        return 0

    def fetch(name: str, size: int | None):
        if args.backend == "gsutil":
            return gcs.download(name, raw, size, use_gsutil=True)
        try:
            return gcs.download(name, raw, size)
        except Exception:
            if args.backend == "auto" and gcs.gsutil_available():
                return gcs.download(name, raw, size, use_gsutil=True)
            raise

    failed = []
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(fetch, n, s): n for n, s in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((futs[fut], e))
            if i % 50 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)} (failed so far: {len(failed)})", flush=True)
    for n, e in failed:
        print(f"FAILED {n}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
