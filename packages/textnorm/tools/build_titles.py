"""Regenerate src/textnorm/data/sefaria_titles.json from Sefaria's public index.

Usage: python tools/build_titles.py path/to/sefaria_api_index.json
(download it with: curl https://www.sefaria.org/api/index)

Only canonical *titles* (identifiers) are extracted, no texts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def walk(node, path, out):
    if isinstance(node, list):
        for x in node:
            walk(x, path, out)
    elif "title" in node and "contents" not in node:
        out.append(("/".join(path), node["title"], node.get("heTitle", "")))
    else:
        for x in node.get("contents", []):
            walk(x, path + [node.get("category", "")], out)


def main(src: str) -> None:
    tree = json.loads(Path(src).read_text(encoding="utf8"))
    leaves: list[tuple[str, str, str]] = []
    walk(tree, [], leaves)
    seder = ("Zeraim", "Moed", "Nashim", "Nezikin", "Kodashim", "Tahorot")
    bavli, mishnah, rambam = [], [], []
    for p, en, he in leaves:
        if p in {f"Talmud/Bavli/Seder {s}" for s in seder}:
            bavli.append({"en": en, "he": he})
        elif p in {f"Mishnah/Seder {s}" for s in seder}:
            mishnah.append({"en": en, "he": he.removeprefix("משנה ")})
        elif p.startswith("Halakhah/Mishneh Torah/Sefer ") and p.count("/") == 2:
            if he.startswith("משנה תורה, הלכות "):
                rambam.append({"en": en, "he": he.removeprefix("משנה תורה, הלכות ")})
    data = {
        "_comment": "Canonical title identifiers from Sefaria's public index API; no texts.",
        "bavli": bavli,
        "mishnah": mishnah,
        "rambam": rambam,
        "shulchan_arukh": {
            "OC": "Shulchan Arukh, Orach Chayim",
            "YD": "Shulchan Arukh, Yoreh De'ah",
            "EH": "Shulchan Arukh, Even HaEzer",
            "CM": "Shulchan Arukh, Choshen Mishpat",
        },
        "mishnah_berurah": "Mishnah Berurah",
    }
    out = Path(__file__).parent.parent / "src/textnorm/data/sefaria_titles.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf8")
    print(len(bavli), len(mishnah), len(rambam))


if __name__ == "__main__":
    main(sys.argv[1])
