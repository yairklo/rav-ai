"""Walk a Sefaria version JSON using its schema and yield canonical text units.

Canonical ids follow Sefaria's ref format: ``<Title>[, <Node title>...] <addr>:<addr>``,
e.g. ``Shulchan Arukh, Orach Chayim 1:1`` or ``Berakhot 2a:3``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

_TAG_RE = re.compile(r"<[^>]*>")


@dataclass
class Unit:
    ref: str
    node_path: tuple[str, ...]  # complex-schema node titles ("" parts omitted)
    labels: tuple[str, ...]  # human addresses, e.g. ("1", "1") or ("2a", "3")
    indices: tuple[int, ...]  # 0-based array positions
    section_names: tuple[str, ...]
    text: str  # raw, byte-for-byte as in the export
    node_index: int = 0


@dataclass
class _Leaf:
    node_path: tuple[str, ...]
    node: dict
    node_index: int = 0
    sub: list = field(default_factory=list)


def address_label(address_type: str, index: int, starting: int = 0) -> str:
    """Label for the `index`-th (0-based) element of an array with this address type."""
    if address_type == "Talmud":
        n = index + starting
        return f"{n // 2 + 1}{'ab'[n % 2]}"
    return str(index + 1 + starting)


def _node_title(node: dict) -> str:
    if node.get("default"):
        return ""
    return node.get("title") or node.get("key") or ""


def leaves(schema: dict, prefix: tuple[str, ...] = ()) -> Iterator[_Leaf]:
    """Yield JaggedArray leaves of the schema tree in document order."""
    counter = [0]

    def rec(node: dict, path: tuple[str, ...]):
        if "nodes" in node:
            for child in node["nodes"]:
                t = _node_title(child)
                yield from rec(child, path + ((t,) if t else ()))
        else:
            yield _Leaf(path, node, counter[0])
            counter[0] += 1

    # root title is added by caller; root node itself contributes no extra path part
    yield from rec(schema, prefix)


def _lookup_text(text, node: dict, path_titles: tuple[str, ...], is_root_leaf: bool):
    """Find the text subtree for a leaf within the version's `text` field."""
    if is_root_leaf:
        return text
    cur = text
    for part in path_titles:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    if isinstance(cur, dict):  # default node nested under its parent
        cur = cur.get("")
    return cur


def iter_units(book_title: str, schema: dict, text) -> Iterator[Unit]:
    """Yield a Unit for every non-empty string in `text`, per `schema`."""
    leaf_list = list(leaves(schema))
    root_is_leaf = "nodes" not in schema
    for leaf in leaf_list:
        node = leaf.node
        subtree = _lookup_text(text, node, leaf.node_path, root_is_leaf)
        if subtree is None and not root_is_leaf and isinstance(text, dict) and not leaf.node_path:
            subtree = text.get("")
        if subtree is None:
            continue
        section_names = tuple(node.get("sectionNames") or ())
        addr_types = list(node.get("addressTypes") or [])
        starting = node.get("startingAddress")
        depth = node.get("depth") or max(len(section_names), 1)
        head = ", ".join((book_title, *leaf.node_path))

        def walk(
            sub,
            d,
            labels,
            idxs,
            head=head,
            leaf=leaf,
            section_names=section_names,
            addr_types=addr_types,
            depth=depth,
            starting=starting,
        ):
            if isinstance(sub, str):
                if sub == "":
                    return
                yield Unit(
                    ref=f"{head} {':'.join(labels)}" if labels else head,
                    node_path=leaf.node_path,
                    labels=tuple(labels),
                    indices=tuple(idxs),
                    section_names=section_names[: len(labels)],
                    text=sub,
                    node_index=leaf.node_index,
                )
            elif isinstance(sub, list):
                level = len(labels)
                at = addr_types[level] if level < len(addr_types) else "Integer"
                start = starting if (level == 0 and isinstance(starting, int)) else 0
                for i, child in enumerate(sub):
                    if child in ("", None, [], {}):
                        continue
                    yield from walk(
                        child, d + 1, [*labels, address_label(at, i, start)], [*idxs, i]
                    )

        yield from walk(subtree, 0, [], [])


def strip_tags(s: str) -> str:
    return _TAG_RE.sub(" ", s)


def count_words(s: str) -> int:
    """Whitespace tokens (after dropping markup) containing at least one letter."""
    return sum(1 for tok in strip_tags(s).split() if any(c.isalpha() for c in tok))


def count_words_in(text) -> int:
    if isinstance(text, str):
        return count_words(text)
    if isinstance(text, list):
        return sum(count_words_in(t) for t in text)
    if isinstance(text, dict):
        return sum(count_words_in(t) for t in text.values())
    return 0
