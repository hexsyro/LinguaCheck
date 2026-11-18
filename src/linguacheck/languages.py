"""Lightweight, dependency-free language detection for LinguaCheck.

Not a translation engine and not a full NLP model — a fast character-script
classifier plus a stopword scorer for Latin-script languages. Good enough to
catch "this text is Spanish but got scored by an English-tuned model".
"""

# RUF001: stopword tables intentionally contain non-Latin glyphs that look like
# Latin letters (e.g. Cyrillic 'es' vs Latin 'c'); they are correct as-is.

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
    "punjabi": "pa", "sinhala": "si", "urdu": "ur", "hungarian": "hu",
}

# ISO-639-3 (plus the common ISO-639-2/B forms such as ``ger``/``slo``) codes
# seen in real model/language metadata. Without this table a naive ``code[:2]``
# fallback turns ``jpn`` into ``jp`` and silently misses the mismatch.
ISO639_3_TO_1: dict[str, str] = {
    "eng": "en", "spa": "es", "fra": "fr", "fre": "fr", "ger": "de", "deu": "de",
    "ita": "it", "por": "pt", "nld": "nl", "dut": "nl",
    "jpn": "ja", "tur": "tr", "swe": "sv", "pol": "pl", "ces": "cs", "cze": "cs",
    "dan": "da", "rus": "ru", "zho": "zh", "chi": "zh", "kor": "ko", "ara": "ar",
    "heb": "he", "ell": "el", "gre": "el", "tha": "th", "hin": "hi", "ben": "bn",
    "urd": "ur", "vie": "vi", "ind": "id", "mal": "ms", "msa": "ms", "ukr": "uk",
    "ron": "ro", "rum": "ro", "hun": "hu", "fin": "fi", "nor": "no", "slo": "sk",
    "slk": "sk", "slv": "sl", "lit": "lt", "lav": "lv", "est": "et", "sqi": "sq",
    "alb": "sq", "isl": "is", "ice": "is", "cat": "ca", "glg": "gl", "cym": "cy",
    "wel": "cy", "gle": "ga", "eus": "eu", "baq": "eu", "afr": "af", "swa": "sw",
    "tgl": "tl", "hrv": "hr", "bul": "bg", "srp": "sr", "mkd": "mk", "bel": "be",
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
    """Rank languages by stopword hits, keeping only languages with evidence.

    Zero-hit languages are not runner-ups: they never had a vote in the first
    place, so including them would inflate the reported margin.
    """
    words = set(_WORD_RE.findall(text.lower()))
    if not words:
        return []
    scored = [(lang, len(words & stopwords)) for lang, stopwords in table.items()]
    scored = [(lang, hits) for lang, hits in scored if hits > 0]
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
    ``confidence`` blends stopword coverage with the best-vs-runner-up gap,
    and ``margin`` is the raw hit difference (0 when no runner-up exists).
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
    """Confidence from stopword coverage and best-vs-runner-up margin.

    ``margin`` is the raw hit difference (0 when there is no runner-up). The
    confidence blend uses a normalized dominance so a sole candidate with 2+
    hits is still trusted, but short texts are capped: a couple of stopwords
    is not much evidence when the whole record is only one or two words.
    """
    words = _WORD_RE.findall(text.lower())
    if not ranked or not words:
        return 0.0, "", 0.0
    _top, top_hits = ranked[0]
    if len(ranked) > 1:
        runner_up, runner_hits = ranked[1]
        margin = float(top_hits - runner_hits)
        dominance = min(1.0, margin / top_hits)
    else:
        runner_up, runner_hits = "", 0
        margin = 0.0
        dominance = 1.0
    if top_hits == 0:
        return 0.15, runner_up, 0.0
    coverage = min(1.0, top_hits / max(1, len(words)) * 2.0)
    confidence = coverage * (0.5 + 0.5 * dominance)
    if len(words) < 4:
        confidence = min(confidence, 0.85)
    return round(min(1.0, confidence), 3), runner_up, round(margin, 3)


_MIXED_SPLIT = re.compile(r"[.?!;\n]+")

_STOPWORDS_BY_CODE = {"latin": LATIN_STOPWORDS, "cyrillic": CYRILLIC_STOPWORDS}

_CJK_SCRIPTS = frozenset({"hiragana", "katakana", "hangul", "cjk"})
_SCRIPT_LANGUAGE: dict[str, str] = {
    **_SCRIPT_LANG,
    "devanagari": "hi",
    "arabic": "ar",
    "hebrew": "he",
    "greek": "el",
    "thai": "th",
}
_MIN_SCRIPT_RUN = 2


def _char_script(char: str) -> str:
    """Script name for one alphabetic character, or '' for separators/emoji."""
    if not char.isalpha():
        return ""
    for name, ranges in _SCRIPTS:
        if _in_ranges(char, ranges):
            return name
    return ""


def _script_runs(text: str) -> list[tuple[str, str, set[str]]]:
    """Group text into consecutive same-family script runs.

    Returns ``(family, chunk, raw_scripts)`` triples. Kana, hangul and Han are
    one ``cjk`` family so a Japanese sentence does not look like a zh/ja switch.
    """
    runs: list[tuple[str, str, set[str]]] = []
    family = ""
    chars: list[str] = []
    raw: set[str] = set()
    for char in text:
        script = _char_script(char)
        current_family = "cjk" if script in _CJK_SCRIPTS else script
        if not current_family:
            # Spaces and punctuation belong to the surrounding run; only a
            # different script actually ends it.
            if family:
                chars.append(char)
            continue
        if current_family == family:
            chars.append(char)
            raw.add(script)
            continue
        if family and chars:
            runs.append((family, "".join(chars), raw))
        family = current_family
        chars = [char]
        raw = {script}
    if family and chars:
        runs.append((family, "".join(chars), raw))
    return runs


def _cjk_language(raw_scripts: set[str]) -> str:
    if raw_scripts & {"hiragana", "katakana"}:
        return "ja"
    if "hangul" in raw_scripts:
        return "ko"
    return "zh"


def _script_switch_languages(text: str) -> list[str]:
    """Languages evidenced by transitions between Unicode scripts in ``text``."""
    found: set[str] = set()
    for family, chunk, raw_scripts in _script_runs(text):
        if len(chunk) < _MIN_SCRIPT_RUN:
            continue
        if family == "cjk":
            found.add(_cjk_language(raw_scripts))
        elif family == "latin":
            ranked = _rank_stopwords(chunk, LATIN_STOPWORDS)
            if ranked and ranked[0][1] >= 1:
                found.add(ranked[0][0])
        elif family == "cyrillic":
            ranked = _rank_stopwords(chunk, CYRILLIC_STOPWORDS)
            found.add(ranked[0][0] if ranked else "ru")
        elif family in _SCRIPT_LANGUAGE:
            found.add(_SCRIPT_LANGUAGE[family])
    return sorted(found)


def _chunked_stopword_switch(text: str, script: str) -> list[str]:
    """Original sentence-chunk stopword fallback for a single script."""
    table = _STOPWORDS_BY_CODE.get(script)
    if table is None:
        counts, _emoji = _script_counts(text)
        if counts.get("latin", 0) > 0:
            table = LATIN_STOPWORDS
        elif counts.get("cyrillic", 0) > 0:
            table = CYRILLIC_STOPWORDS
        else:
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


def detect_code_switching(text: str, script: str = "") -> list[str]:
    """Languages used confidently in different parts of one text.

    Script transitions (Latin/Cyrillic/Greek/Arabic/Devanagari/CJK/…) are the
    primary signal; the sentence-chunk stopword scorer remains as a fallback
    for texts that stay inside one script (``en / es`` in Latin script, etc.).
    """
    languages = _script_switch_languages(text)
    return languages if len(languages) >= 2 else _chunked_stopword_switch(text, script)


def language_distribution(text: str) -> dict[str, float]:
    """Per-language shares of one record (rounding to three decimals).

    Decisive scripts are counted in characters. Latin text is assigned word by
    word: a stopword shared by several languages goes to the record's strongest
    language, so a Spanish sentence isn't quietly reported as 60% Catalan.
    Returns an empty dict when there is no evidence at all.
    """
    if not text or not text.strip():
        return {}
    counts, _emoji = _script_counts(text)
    weights: dict[str, float] = {}

    if counts.get("latin", 0):
        ranked = _rank_stopwords(text, LATIN_STOPWORDS)
        if ranked:
            preferred = ranked[0][0]
            eligible = {lang for lang, hits in ranked if hits >= 2}
            for word in _WORD_RE.findall(text.lower()):
                candidates = [
                    lang
                    for lang, stops in LATIN_STOPWORDS.items()
                    if word in stops and (lang in eligible or lang == preferred)
                ]
                if candidates:
                    lang = preferred if preferred in candidates else candidates[0]
                    weights[lang] = weights.get(lang, 0.0) + 1
        else:
            weights["en"] = weights.get("en", 0.0) + counts["latin"]
    if counts.get("cyrillic", 0):
        weights["ru"] = weights.get("ru", 0.0) + counts["cyrillic"]

    cjk_total = sum(counts.get(script, 0) for script in _CJK_SCRIPTS)
    if cjk_total:
        if counts.get("hiragana", 0) or counts.get("katakana", 0):
            language = "ja"
        elif counts.get("hangul", 0):
            language = "ko"
        else:
            language = "zh"
        weights[language] = weights.get(language, 0.0) + cjk_total
    for script, count in counts.items():
        if script in _CJK_SCRIPTS or script in {"latin", "cyrillic"}:
            continue
        language = _SCRIPT_LANGUAGE.get(script)
        if language:
            weights[language] = weights.get(language, 0.0) + count

    total = sum(weights.values())
    if total <= 0:
        return {}
    ranked = sorted(weights.items(), key=lambda item: -item[1])
    return {lang: round(weight / total, 3) for lang, weight in ranked}


def normalize_lang(value: str) -> str:
    """Map a messy language label (``English``, ``eng``, ``pt-BR``) to a code.

    Language names, ISO-639-1 codes and the common ISO-639-3/2 codes are all
    understood before the last-resort two-letter truncation.
    """
    code = re.sub(r"[^a-zA-Z]", "", value).strip().lower()
    if not code:
        return "unknown"
    if code in LANG_NAMES:
        return LANG_NAMES[code]
    if len(code) == 2:
        return code
    if code in ISO639_3_TO_1:
        return ISO639_3_TO_1[code]
    return code[:2] if len(code) >= 2 else "unknown"
