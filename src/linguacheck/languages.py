"""Lightweight, dependency-free language detection for LinguaCheck.

Not a translation engine and not a full NLP model — a fast character-script
classifier plus a stopword scorer for Latin-script languages. Good enough to
catch "this text is Spanish but got scored by an English-tuned model".
"""

# RUF001: stopword tables intentionally contain non-Latin glyphs that look like
# Latin letters (e.g. Cyrillic 'es' vs Latin 'c'); they are correct as-is.
# ruff: noqa: RUF001

from __future__ import annotations

import re


def _stop(words: str) -> frozenset[str]:
    return frozenset(words.split())


LATIN_STOPWORDS: dict[str, frozenset[str]] = {
    "en": _stop("the and of to that is in you for it this with are was be not have has on as at but do from your"),
    "es": _stop("el la los las de del que y en un una es no por con para se lo como son ha su mas pero"),
    "fr": _stop("le la les de des du et est un une que pas pour avec dans sur ce se je il elle nous vous ont au"),
    "de": _stop("der die das und ist nicht ein eine mit auf fur zu den dem des im ich du er sie es wir sind als von"),
    "it": _stop("il lo la gli le di del e un uno una che non per con su in sono ho ha i questo ma se"),
    "pt": _stop("o os a as de do da dos das e um uma que nao para com por em sao foi mas como mais se"),
    "nl": _stop("de het een en van is op met voor dat te zijn niet ook maar aan in om we hij zij er bij als"),
    "sv": _stop("och att det som en ar av for till med pa inte den jag du han hon vi de om men fran"),
    "pl": _stop("i w nie to na sie z jest o ze do co ja ty my ale dla po jak tez"),
    "id": _stop("yang dan di itu ini untuk dengan adalah tidak ke dari saya kamu juga pada akan sudah"),
    "tr": _stop("ve bir bu icin ile cok de da o ben sen biz siz olarak var yok ne mi mi"),
    "vi": _stop("va la cua khong co de trong nay do voi nguoi toi ban chung duoc cung nhu o ra"),
    "ro": _stop("si de la in un o este nu cu pe ca mai pentru se dar sau eu tu noi voi"),
    "cs": _stop("a je se na ze to ve z s o pro do by jako ale tak uz jen ne"),
    "da": _stop("og at det en er til pa jeg du han hun vi de ikke med for som af"),
    "no": _stop("og at det en er til pa jeg du han hun vi de ikke med for som av"),
    "fi": _stop("ja on ei se etta kun myos tama mina sina han me te he olla oli mutta vaan"),
    "el": _stop("kai to tou stis se gia me tha oi sto apo mia auti ti den sto"),
    # Low-resource coverage: languages budget-tier sentiment models rarely
    # support, added so a mismatch is still caught instead of scored blind.
    "ca": _stop("el la els les de del que i en un una es no per amb se lo com son ha"),
    "gl": _stop("o a os as de do da que e en un unha non por con para se como son"),
    "cy": _stop("y yr a yn i o ac mae bod wedi ar gyda gan am fel ond neu"),
    "ga": _stop("an na agus ar is go do le se sin nach me tu se si muid sibh siad"),
    "eu": _stop("eta da ez bat bere baina edo hau hori ni zu hura bizi naiz du"),
    "af": _stop("die en van is nie in op vir met het wat ek jy hy sy ons hulle"),
    "sw": _stop("na ya kwa katika ni wa hii hiyo lakini au pia sana kama hata"),
    "tl": _stop("ang ng mga sa ay na at ito iyon para hindi o kung bilang naman"),
    "hr": _stop("je i u na se da za od su kao ali ne ovo koji sto sa bi"),
    "sk": _stop("je a v na sa ze to do pre ako ale tak uz len nie ktory co"),
    "sl": _stop("je in v na se da za od so kot ali ne to ki bi tudi ze"),
    "lt": _stop("yra ir kad bet ar tai kaip jis ji o ne su apie tik dar"),
    "lv": _stop("ir un kad bet ar tas ka ne su par ari vel jau tikai"),
    "et": _stop("on ja ei see et ka aga voi kui ta me ole olema ning"),
    "sq": _stop("dhe eshte nje per me se ne kete ate po nuk jane si nga"),
    "is": _stop("og ad er ekki thetta hvad sem vid a eftir meira en um"),
}

CYRILLIC_STOPWORDS: dict[str, frozenset[str]] = {
    "ru": _stop("и в не на что с по это он она мы вы они как из за для от было есть"),
    "uk": _stop("і в не на що з по це він вона ми ви вони як за для від було є"),
    "bg": _stop("и в не на че с по това той тя ние вие те като от за защото е са"),
    "sr": _stop("и у на се да за од су као али не ово који што би"),
    "mk": _stop("и во на се да за од е што како но али ова со би"),
    "be": _stop("і ў не на што з па гэта ён яна мы вы яны як за для ад"),
}

LANG_NAMES: dict[str, str] = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "arabic": "ar",
    "chinese": "zh", "mandarin": "zh", "japanese": "ja", "korean": "ko",
    "russian": "ru", "turkish": "tr", "hindi": "hi", "polish": "pl",
    "swedish": "sv", "vietnamese": "vi", "indonesian": "id", "thai": "th",
    "hebrew": "he", "greek": "el", "ukrainian": "uk", "czech": "cs",
    "romanian": "ro", "danish": "da", "norwegian": "no", "finnish": "fi",
    "bulgarian": "bg", "catalan": "ca", "welsh": "cy", "irish": "ga",
    "basque": "eu", "galician": "gl", "afrikaans": "af", "swahili": "sw",
    "tagalog": "tl", "filipino": "tl", "croatian": "hr", "slovak": "sk",
    "slovenian": "sl", "lithuanian": "lt", "latvian": "lv", "estonian": "et",
    "albanian": "sq", "icelandic": "is", "serbian": "sr", "macedonian": "mk",
    "belarusian": "be", "bengali": "bn", "tamil": "ta",
    "telugu": "te", "gujarati": "gu", "kannada": "kn", "malayalam": "ml",
    "punjabi": "pa", "sinhala": "si",
}

# Languages treated as low-resource in v1: they are supported by the detector
# but are commonly missing from off-the-shelf sentiment models, so a mismatch
# carries extra weight and the summary reports coverage explicitly.
LOW_RESOURCE_LANGS: frozenset[str] = frozenset({
    "ca", "cy", "ga", "eu", "gl", "af", "sw", "tl", "hr", "sk", "sl", "lt",
    "lv", "et", "sq", "is", "sr", "mk", "be", "uk", "bg", "el", "vi", "id",
    "th", "hi", "bn", "ta", "te", "gu", "kn", "ml", "pa", "si",
})


def is_low_resource(code: str) -> bool:
    """True when a language code is in the extended low-resource coverage set."""
    return code in LOW_RESOURCE_LANGS

_SCRIPTS: list[tuple[str, list[tuple[int, int]]]] = [
    ("hangul", [(0xAC00, 0xD7AF), (0x1100, 0x11FF)]),
    ("hiragana", [(0x3040, 0x309F)]),
    ("katakana", [(0x30A0, 0x30FF)]),
    ("cjk", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]),
    ("cyrillic", [(0x0400, 0x04FF)]),
    ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)]),
    ("hebrew", [(0x0590, 0x05FF)]),
    ("greek", [(0x0370, 0x03FF)]),
    ("thai", [(0x0E00, 0x0E7F)]),
    ("devanagari", [(0x0900, 0x097F)]),
    # Low-resource Indic + Sinhala scripts (script evidence is decisive on its
    # own, so these need no stopword tables).
    ("bengali", [(0x0980, 0x09FF)]),
    ("gurmukhi", [(0x0A00, 0x0A7F)]),
    ("gujarati", [(0x0A80, 0x0AFF)]),
    ("tamil", [(0x0B80, 0x0BFF)]),
    ("telugu", [(0x0C00, 0x0C7F)]),
    ("kannada", [(0x0C80, 0x0CFF)]),
    ("malayalam", [(0x0D00, 0x0D7F)]),
    ("sinhala", [(0x0D80, 0x0DFF)]),
    ("latin", [(0x0041, 0x007A), (0x00C0, 0x024F)]),
]

# Decisive script -> language code for the low-resource scripts above.
_SCRIPT_LANG: dict[str, str] = {
    "bengali": "bn",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "sinhala": "si",
}

_EMOJI: list[tuple[int, int]] = [
    (0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF), (0x2600, 0x27BF), (0xFE00, 0xFE0F),
]

_WORD_RE = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)


def _in_ranges(char: str, ranges: list[tuple[int, int]]) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in ranges)


def _script_counts(text: str) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    emoji = 0
    for char in text:
        if _in_ranges(char, _EMOJI):
            emoji += 1
            continue
        for name, ranges in _SCRIPTS:
            if _in_ranges(char, ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    return counts, emoji


def _score_stopwords(text: str, table: dict[str, frozenset[str]]) -> str:
    ranked = _rank_stopwords(text, table)
    return ranked[0][0] if ranked else ""


def _rank_stopwords(text: str, table: dict[str, frozenset[str]]) -> list[tuple[str, int]]:
    words = set(_WORD_RE.findall(text.lower()))
    if not words:
        return []
    scored = [(lang, len(words & stopwords)) for lang, stopwords in table.items()]
    scored.sort(key=lambda item: -item[1])
    return scored


def detect_language(text: str) -> tuple[str, str]:
    """Return ``(iso639_code, script_name)`` for a piece of text.

    ``script_name`` is one of ``latin``, ``cyrillic``, ``cjk``, ``arabic``,
    ``hebrew``, ``greek``, ``thai``, ``devanagari``, ``emoji`` or ``none``.
    Returns ``("unknown", script)`` when the script can't be attributed.
    """
    code, script, _confidence, _runner_up, _margin = detect_language_scored(text)
    return code, script


def detect_language_scored(text: str) -> tuple[str, str, float, str, float]:
    """Detect language with a calibrated confidence score.

    Returns ``(code, script, confidence, runner_up, margin)`` where
    ``confidence`` blends stopword coverage with the margin between the best
    and second-best candidate, and ``margin`` is the normalized score gap.
    Script-level evidence (CJK, Arabic, …) carries high confidence on its own.
    """
    if not text or not text.strip():
        return "unknown", "none", 0.0, "", 0.0

    counts, emoji = _script_counts(text)
    if counts:
        hiragana = counts.get("hiragana", 0) + counts.get("katakana", 0)
        if hiragana > 0:
            return "ja", "cjk", 0.9, "zh", 1.0
        if counts.get("hangul", 0) > 0:
            return "ko", "cjk", 0.9, "zh", 1.0
        if counts.get("cjk", 0) > 0:
            return "zh", "cjk", 0.85, "ja", 1.0
        if counts.get("cyrillic", 0) > 0:
            ranked = _rank_stopwords(text, CYRILLIC_STOPWORDS)
            code = ranked[0][0] if ranked and ranked[0][1] > 0 else "ru"
            conf, runner_up, margin = _latin_confidence(text, ranked)
            return code, "cyrillic", max(conf, 0.7), runner_up, margin
        if counts.get("arabic", 0) > 0:
            return "ar", "arabic", 0.85, "", 1.0
        if counts.get("hebrew", 0) > 0:
            return "he", "hebrew", 0.85, "", 1.0
        if counts.get("greek", 0) > 0:
            return "el", "greek", 0.85, "", 1.0
        if counts.get("thai", 0) > 0:
            return "th", "thai", 0.85, "", 1.0
        if counts.get("devanagari", 0) > 0:
            return "hi", "devanagari", 0.85, "", 1.0
        for script_name, lang_code in _SCRIPT_LANG.items():
            if counts.get(script_name, 0) > 0:
                return lang_code, script_name, 0.85, "", 1.0
        if counts.get("latin", 0) > 0:
            ranked = _rank_stopwords(text, LATIN_STOPWORDS)
            code = ranked[0][0] if ranked and ranked[0][1] > 0 else "en"
            conf, runner_up, margin = _latin_confidence(text, ranked)
            return code, "latin", conf, runner_up, margin

    if emoji > 0:
        return "unknown", "emoji", 0.0, "", 0.0
    return "unknown", "none", 0.0, "", 0.0


def _latin_confidence(
    text: str, ranked: list[tuple[str, int]]
) -> tuple[float, str, float]:
    """Confidence from stopword coverage and best-vs-runner-up margin."""
    words = _WORD_RE.findall(text.lower())
    if not ranked or not words:
        return 0.0, "", 0.0
    _top, top_hits = ranked[0]
    runner_up, runner_hits = ranked[1] if len(ranked) > 1 else ("", 0)
    if top_hits == 0:
        return 0.15, runner_up, 0.0
    margin = (top_hits - runner_hits) / top_hits
    coverage = min(1.0, top_hits / max(len(words), 1) * 4.0)
    confidence = coverage * (0.5 + 0.5 * margin)
    return round(min(1.0, confidence), 3), runner_up, round(margin, 3)


_MIXED_SPLIT = re.compile(r"[.?!;\n]+")

_STOPWORDS_BY_CODE = {"latin": LATIN_STOPWORDS, "cyrillic": CYRILLIC_STOPWORDS}


def detect_code_switching(text: str, script: str) -> list[str]:
    """Languages used confidently in different sentences of one text.

    Returns the sorted language codes when two or more languages each score
    2+ stopword hits in different sentences — the signal that a single-language
    sentiment model is insufficient. Latin and Cyrillic scripts only in v1.
    """
    table = _STOPWORDS_BY_CODE.get(script)
    if table is None:
        return []
    chunks = [c for c in _MIXED_SPLIT.split(text) if len(c.split()) >= 3]
    if len(chunks) < 2:
        return []
    found: set[str] = set()
    for chunk in chunks:
        ranked = _rank_stopwords(chunk, table)
        # A chunk votes only on a decisive 2+ stopword hits.
        if ranked and ranked[0][1] >= 2:
            found.add(ranked[0][0])
    return sorted(found) if len(found) >= 2 else []


def normalize_lang(value: str) -> str:
    """Map a messy language label (``English``, ``eng``, ``pt-BR``) to a code."""
    code = re.sub(r"[^a-zA-Z]", "", value).strip().lower()
    if not code:
        return "unknown"
    if code in LANG_NAMES:
        return LANG_NAMES[code]
    if len(code) >= 3 and code in {"eng", "spa", "fra", "ger", "ita", "por", "nld", "deu"}:
        return {
            "eng": "en", "spa": "es", "fra": "fr", "ger": "de", "deu": "de",
            "ita": "it", "por": "pt", "nld": "nl",
        }[code]
    return code[:2] if len(code) >= 2 else "unknown"
