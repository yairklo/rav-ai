"""textnorm: normalization, acronym detection and citation parsing for rabbinic Hebrew/Aramaic."""

from .acronyms import AcronymCandidate, Expansion, find_acronyms, load_dictionary, rank_candidates
from .citations import Citation, find_citations
from .gematria import parse_numeral, to_gematria
from .markup import strip_sefaria_markup
from .normalize import NormResult, normalize
from .offsets import decode_spans, encode_spans

__all__ = [
    "AcronymCandidate", "Citation", "Expansion", "NormResult", "find_acronyms",
    "find_citations", "load_dictionary", "normalize", "parse_numeral", "rank_candidates",
    "to_gematria", "decode_spans", "encode_spans", "strip_sefaria_markup",
]  # fmt: skip
