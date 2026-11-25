"""Compact serialization of an offset map (the ``spans`` of a ``NormResult``).

Format: zlib( varint(zigzag(start - prev_start)), varint(end - start) per normalized char ).
Typical text (mostly 1:1 spans) compresses to a fraction of a byte per character.
"""

from __future__ import annotations

import zlib

Spans = tuple[tuple[int, int], ...]


def _put(buf: bytearray, v: int) -> None:
    while v >= 0x80:
        buf.append((v & 0x7F) | 0x80)
        v >>= 7
    buf.append(v)


def encode_spans(spans: Spans) -> bytes:
    buf = bytearray()
    prev = 0
    for s, e in spans:
        d = s - prev
        _put(buf, d << 1 if d >= 0 else ((-d) << 1) - 1)  # zigzag
        _put(buf, e - s)
        prev = s
    return zlib.compress(bytes(buf), 6)


def decode_spans(data: bytes) -> Spans:
    vals: list[int] = []
    cur = shift = 0
    for b in zlib.decompress(data):
        cur |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            vals.append(cur)
            cur = shift = 0
    out: list[tuple[int, int]] = []
    prev = 0
    for k in range(0, len(vals), 2):
        z, ln = vals[k], vals[k + 1]
        d = (z >> 1) if not z & 1 else -((z + 1) >> 1)
        s = prev + d
        out.append((s, s + ln))
        prev = s
    return tuple(out)
