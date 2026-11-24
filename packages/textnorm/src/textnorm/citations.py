"""Explicit citation detection ("שבת כא ע"ב", "או"ח סי' תרע"א ס"ב", ...) -> Sefaria refs.

Parsing runs on level-A normalized text (so niqqud, quote variants and final letters do not
matter) and spans are mapped back to the original with the offset map.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from .gematria import parse_numeral
from .normalize import normalize


@dataclass(frozen=True)
class Citation:
    span: tuple[int, int]  # in the ORIGINAL text
    text: str
    id: str | None  # Sefaria-style ref, e.g. "Shabbat 21b"; None if it cannot be resolved
    confidence: float  # 0..1 heuristic
    kind: str  # bavli | mishnah | shulchan_arukh | mishnah_berurah | rambam | daf
    parts: dict = field(default_factory=dict, compare=False)


def _f(s: str) -> str:
    return normalize(s, "A").text


@lru_cache(maxsize=1)
def _tables():
    d = json.loads(
        resources.files("textnorm").joinpath("data/sefaria_titles.json").read_text("utf8")
    )
    bavli = {_f(x["he"]): x["en"] for x in d["bavli"]}
    for abbr, en in {
        'ב"ק': "Bava Kamma", 'ב"מ': "Bava Metzia", 'ב"ב': "Bava Batra", 'ע"ז': "Avodah Zarah",
        'ר"ה': "Rosh Hashanah", 'מו"ק': "Moed Katan", "מועד קטן": "Moed Katan",
        "שבועות": "Shevuot", "ביצה": "Beitzah", "בכורות": "Bekhorot", "מכות": "Makkot",
        "עירובין": "Eruvin", "תענית": "Taanit", "חגיגה": "Chagigah", "כריתות": "Keritot",
        "מעילה": "Meilah", "חולין": "Chullin", "ערכין": "Arakhin", "נידה": "Niddah",
        "קידושין": "Kiddushin", "מנחות": "Menachot",
    }.items():  # fmt: skip
        if en in {x["en"] for x in d["bavli"]}:
            bavli.setdefault(_f(abbr), en)
    mishnah = {_f(x["he"]): x["en"] for x in d["mishnah"]}
    rambam = {_f(x["he"]): x["en"] for x in d["rambam"]}
    rambam_alias = {
        "חנוכה": "Scroll of Esther and Hanukkah", "מגילה": "Scroll of Esther and Hanukkah",
        "שבת": "Sabbath", "שבתות": "Sabbath", "תשובה": "Repentance",
    }  # fmt: skip
    en_by_title = {x["en"]: x["en"] for x in d["rambam"]}
    for k, v in rambam_alias.items():
        if "Mishneh Torah, " + v in en_by_title:
            rambam.setdefault(_f(k), "Mishneh Torah, " + v)
    sa = {
        'או"ח': "OC", "אורח חיים": "OC", 'יו"ד': "YD", "יורה דעה": "YD",
        'אה"ע': "EH", "אבן העזר": "EH", 'חו"מ': "CM", "חושן משפט": "CM",
    }  # fmt: skip
    return bavli, mishnah, rambam, {_f(k): v for k, v in sa.items()}, d


_NUM = r"(?:\d+|[א-ת]+(?:\"[א-ת]+)?|[א-ט]'[א-ת\"]+)"
_PRE_NUM = re.compile(r"^([א-ת])[\"']([א-ת]+)$")
_AMUD = {'ע"א': "a", 'ע"ב': "b", "עא": "a", "עב": "b", "ע'א": "a", "ע'ב": "b"}


_OTHER_WORKS = {
    _f(w)
    for w in (
        "טור", "הטור", 'רמ"א', 'ש"ך', 'ט"ז', 'ב"י', 'מג"א', 'פמ"ג', 'ביה"ל', 'ערוה"ש', 'ב"ח',
        'ד"מ', 'באה"ט', 'שעה"צ', "בית יוסף", "דרכי משה", "באר היטב", "ביאור הלכה", "שער הציון",
        "מגן אברהם", "טורי זהב", "שפתי כהן", "ערוך השולחן", "פרי מגדים", "חיי אדם", "כף החיים",
        "שערי תשובה", "בית חדש", 'חק"י',
    )
}  # fmt: skip


class _Tok:
    __slots__ = ("raw", "t", "start", "end", "tail")

    def __init__(self, raw: str, start: int):
        self.raw = raw
        core = raw.lstrip("([{")
        start += len(raw) - len(core)
        core = core.rstrip(")]},;")
        tail = ""
        if core and core[-1] in ".:" and len(core) > 1:
            tail, core = core[-1], core[:-1]
        self.t, self.tail = core, tail
        self.start, self.end = start, start + len(core) + len(tail)


def _tokenize(s: str) -> list[_Tok]:
    return [_Tok(m.group(), m.start()) for m in re.finditer(r"\S+", s)]


def _num(t: str) -> int | None:
    return parse_numeral(t) if t else None


class _Parser:
    def __init__(self, text: str, context: str | None):
        self.norm = normalize(text, "A")
        self.toks = _tokenize(self.norm.text)
        self.context = context
        self.bavli, self.mishnah, self.rambam, self.sa, self.raw = _tables()

    # -- helpers ------------------------------------------------------------------
    def tok(self, i: int) -> str:
        return self.toks[i].t if i < len(self.toks) else ""

    def lookup_title(self, table: dict, i: int, prefixed: bool = True):
        """Longest multiword match at token i -> (value, n_tokens)."""
        for n in (3, 2, 1):
            if i + n <= len(self.toks):
                w = " ".join(self.tok(i + k) for k in range(n))
                if w in table:
                    return table[w], n
                if prefixed and n == 1 and len(w) > 2 and w[0] in "בלמוהש" and w[1:] in table:
                    return table[w[1:]], n
        return None

    def work(self, i: int, names: tuple[str, ...]) -> str | None:
        """Token at i if it is one of ``names``, allowing 1-2 clitic prefix letters (וב, ה...)."""
        t = self.tok(i)
        for n in range(3):
            if t[n:] in names and all(c in "ובהכלמש" for c in t[:n]):
                return t[n:]
        return None

    def number(self, i: int):
        """Numeral at token i (gematria/digits) -> value or None."""
        return _num(self.tok(i))

    def chapter(self, i: int):
        """פרק ג | פ"ג -> (n, ntok)."""
        t = self.tok(i)
        if t in ("פרק", "פ'", "פ"):
            v = self.number(i + 1)
            return (v, 2) if v else None
        m = _PRE_NUM.match(t)
        if m and m.group(1) == "פ":
            v = _num(m.group(2))
            return (v, 1) if v else None
        return None

    def prefixed(self, i: int, letters: str, words: tuple[str, ...]):
        """ה"א / הלכה א ; מ"ד / משנה ד ; ס"ב / סעיף ב -> (n, ntok, via_abbrev)."""
        t = self.tok(i)
        if t in words:
            v = self.number(i + 1)
            return (v, 2, False) if v else None
        m = _PRE_NUM.match(t)
        if m and m.group(1) in letters:
            v = _num(m.group(2))
            return (v, 1, True) if v else None
        if t.startswith("הל") and "הלכה" in words:
            m2 = re.match(r"^הל[\"']([א-ת]+)$", t)
            if m2 and _num(m2.group(1)):
                return _num(m2.group(1)), 1, True
        return None

    # -- individual grammars ------------------------------------------------------
    def bavli_ref(self, i: int):
        j = i
        had_masechet = False
        if self.tok(j) in ("מסכת", "במסכת", "מס'", "מסכתא"):
            j += 1
            had_masechet = True
        hit = self.lookup_title(self.bavli, j)
        if not hit:
            return None
        title, n = hit
        k = j + n
        had_daf = False
        if self.tok(k) in ("דפ", "ד'", "דפימ"):
            k += 1
            had_daf = True
        daf = self.number(k)
        if not daf or daf < 2 or daf > 176:
            return None
        dt = self.toks[k]
        k += 1
        amud, explicit = None, False
        if dt.tail == ":":
            amud, explicit = "b", True
        elif dt.tail == ".":
            amud = "a"
        elif self.tok(k) in _AMUD:
            amud, explicit = _AMUD[self.tok(k)], True
            k += 1
        elif self.tok(k) == "עמוד" and self.tok(k + 1) in ("א", "ב"):
            amud, explicit = "a" if self.tok(k + 1) == "א" else "b", True
            k += 2
        if re.match(r"^פ[\"']", dt.t) and not (had_daf or explicit):
            return None  # פ"ב is "chapter 2" (or 82 only with דף / explicit amud)
        conf = 0.6
        conf += 0.25 if explicit else (0.05 if amud else 0.0)
        conf += 0.1 if had_daf else 0.0
        conf += 0.05 if had_masechet else 0.0
        ref = f"{title} {daf}{amud or ''}"
        return (i, k, ref, min(conf, 0.98), "bavli",
                {"tractate": title, "daf": daf, "amud": amud})  # fmt: skip

    def bare_daf(self, i: int):
        if self.tok(i) not in ("דפ", "ד'"):
            return None
        daf = self.number(i + 1)
        if not daf or daf < 2 or daf > 176:
            return None
        k = i + 2
        dt = self.toks[i + 1]
        amud = None
        if dt.tail == ":":
            amud = "b"
        elif dt.tail == ".":
            amud = "a"
        elif self.tok(k) in _AMUD:
            amud = _AMUD[self.tok(k)]
            k += 1
        ref = f"{self.context} {daf}{amud or ''}" if self.context else None
        return (i, k, ref, 0.7 if ref else 0.4, "daf", {"daf": daf, "amud": amud})

    def mishnah_ref(self, i: int):
        j = i
        explicit = False
        if self.tok(j) in ("משנה", "מתני'", "מתניתינ", "במשנה"):
            j += 1
            explicit = True
        if self.tok(j) in ("מסכת", "במסכת"):
            j += 1
        hit = self.lookup_title(self.mishnah, j)
        if not hit:
            return None
        title, n = hit
        k = j + n
        ch = self.chapter(k)
        if not ch:
            return None
        k += ch[1]
        mi = self.prefixed(k, "מ", ("משנה",))
        if not mi:
            return None
        k += mi[1]
        return (i, k, f"{title} {ch[0]}:{mi[0]}", 0.9 if explicit else 0.6, "mishnah",
                {"tractate": title, "chapter": ch[0], "mishnah": mi[0]})  # fmt: skip

    def sa_like(self, i: int, other: bool = False):
        j = i
        work = None
        if self.work(j, ("שולחנ", "שלחנ")) and self.tok(j + 1) == "ערוכ":
            j, work = j + 2, "sa"
        elif self.work(j, ('שו"ע',)):
            j, work = j + 1, "sa"
        elif self.work(j, ("משנה",)) and self.tok(j + 1) == "ברורה":
            j, work = j + 2, "mb"
        elif self.work(j, ('משנ"ב', 'מ"ב')):
            j, work = j + 1, "mb"
        part = None
        for n in (2, 1):
            w = " ".join(self.tok(j + k) for k in range(n))
            if w in self.sa:
                part, j = self.sa[w], j + n
                break
        if (
            not other
            and work is None
            and part is None
            and not (self.context and self.context.startswith("Shulchan Arukh,"))
        ):
            return None
        if self.tok(j) not in ("סי'", "סימנ"):
            return None
        siman = self.number(j + 1)
        if not siman or siman > 697:
            return None
        k = j + 2
        parts: dict = {"siman": siman}
        seif = sk = None
        for _ in range(2):  # סעיף and/or ס"ק, in either order
            if sk is None and self.tok(k) in ('ס"ק', "סק") and self.number(k + 1):
                sk, k = self.number(k + 1), k + 2
            elif seif is None and self.tok(k) not in ('ס"ק', "סק"):
                r = self.prefixed(k, "ס", ("סעיפ", "ס'"))
                if r:
                    seif, k = r[0], k + r[1]
        if other:
            return (i, k, None, 0.5, "other_work", {**parts, "seif": seif, "seif_katan": sk})
        if work == "mb":
            # Sefaria addresses the Mishnah Berurah as siman:se'if-katan. A plain se'if is
            # kept in the id too (as requested) but at lower confidence: it may not be the
            # same number as the se'if katan.
            last = sk or seif
            ref = f"Mishnah Berurah {siman}" + (f":{last}" if last else "")
            conf = 0.9 if sk else (0.6 if seif else 0.7)
            kind = "mishnah_berurah"
            parts.update(seif_katan=sk, seif=seif)
        else:
            hp = part or None
            if hp is None:
                if not self.context or not self.context.startswith("Shulchan Arukh,"):
                    return None
                title, conf0 = self.context, 0.6
            else:
                title, conf0 = self.raw["shulchan_arukh"][hp], 0.8
            ref = f"{title} {siman}" + (f":{seif}" if seif else "")
            conf = conf0 + (0.1 if seif else 0) + (0.05 if work else 0)
            kind = "shulchan_arukh"
            parts.update(seif=seif, part=hp)
        return (i, k, ref, min(conf, 0.97), kind, parts)

    def other_work(self, i: int):
        """רמ"א / ש"ך / ט"ז / טור ... סי' X: a work we do not resolve -> id None, not שו"ע."""
        for n in (2, 1):
            w = " ".join(self.tok(i + k) for k in range(n))
            for pre in range(3):
                if w[pre:] in _OTHER_WORKS and all(c in "ובהכלמש" for c in w[:pre]):
                    r = self.sa_like(i + n, other=True)
                    if r:
                        return (i, r[1], None, 0.5, "other_work", {**r[5], "work": w[pre:]})
                    return None
        return None

    def yerushalmi_ref(self, i: int):
        """ירושלמי <tractate> [פרק/הלכה/דף]: recognized but not resolved (id None)."""
        j = i
        if self.tok(j) == "תלמוד" and self.work(j + 1, ("ירושלמי",)):
            j += 1
        if not self.work(j, ("ירושלמי", "ירו'", "ירוש'")):
            return None
        j += 1
        if self.tok(j) in ("מסכת", "במסכת"):
            j += 1
        hit = self.lookup_title(self.mishnah, j) or self.lookup_title(self.bavli, j)
        if not hit:
            return None
        k = j + hit[1]
        ch = self.chapter(k)
        if ch:
            k += ch[1]
            hl = self.prefixed(k, "ה", ("הלכה", "הל'"))
            if hl:
                k += hl[1]
        else:
            if self.tok(k) in ("דפ", "ד'"):
                k += 1
            if self.number(k):
                k += 1
                if self.tok(k) in _AMUD:
                    k += 1
        return (i, k, None, 0.5, "yerushalmi", {"tractate": hit[0]})

    def rambam_ref(self, i: int):
        j = i
        named = False
        if self.work(j, ('רמב"מ', "משנה")):
            if self.work(j, ("משנה",)):
                if self.tok(j + 1) != "תורה":
                    return None
                j += 1
            j += 1
            named = True
        had_hal = False
        if self.tok(j) in ("הל'", "הלכות", "הלכה"):
            j += 1
            had_hal = True
        if not named and not had_hal:
            return None
        hit = self.lookup_title(self.rambam, j, prefixed=False)
        if not hit:
            return None
        title, n = hit
        k = j + n
        ch = self.chapter(k)
        ref = title
        parts: dict = {"book": title}
        conf = 0.7 if named else 0.5
        if ch:
            k += ch[1]
            hl = self.prefixed(k, "ה", ("הלכה", "הל'"))
            parts["chapter"] = ch[0]
            if hl:
                k += hl[1]
                ref += f" {ch[0]}:{hl[0]}"
                parts["halakhah"] = hl[0]
                conf = 0.92 if named else 0.75
            else:
                ref += f" {ch[0]}"
                conf += 0.1
        return (i, k, ref, conf, "rambam", parts)

    # -- driver -----------------------------------------------------------------
    def run(self) -> list[Citation]:
        out: list[Citation] = []
        i = 0
        grammars = (
            self.rambam_ref,
            self.sa_like,
            self.other_work,
            self.yerushalmi_ref,
            self.mishnah_ref,
            self.bavli_ref,
            self.bare_daf,
        )
        while i < len(self.toks):
            best = None
            for g in grammars:
                r = g(i)
                if r and (best is None or r[1] > best[1]):
                    best = r
            if best:
                a, b, ref, conf, kind, parts = best
                ns, ne = self.toks[a].start, self.toks[b - 1].end
                s, e = self.norm.map_span(ns, ne)
                out.append(
                    Citation((s, e), self.norm.original[s:e], ref, round(conf, 2), kind, parts)
                )
                i = b
            else:
                i += 1
        return out


def find_citations(text: str, *, context: str | None = None) -> list[Citation]:
    """Find explicit citations in ``text``.

    ``context`` is an optional Sefaria title (e.g. "Shabbat", or "Shulchan Arukh, Orach Chayim")
    used to resolve bare references such as "דף כ"א" or "סי' תרע"א ס"ב". Without it, bare refs are
    returned with ``id=None``. Spans refer to the original ``text``.
    """
    return _Parser(text, context).run()
