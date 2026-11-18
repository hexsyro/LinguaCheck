"""Detector-level tests: language ID, normalization, code-switching."""

from __future__ import annotations

from linguacheck.languages import (
    detect_code_switching,
    detect_language,
    detect_language_scored,
    is_low_resource,
    language_distribution,
    normalize_lang,
)


def test_detect_english():
    code, script = detect_language("This is the best product I have ever used and it keeps working")
    assert code == "en"
    assert script == "latin"


def test_detect_spanish():
    code, script = detect_language("Esto es lo peor que he comprado y no lo recomiendo para nada")
    assert code == "es"
    assert script == "latin"


def test_detect_french():
    code, _ = detect_language("Bonjour tout le monde, c'est très bien et je l'aime beaucoup")
    assert code == "fr"


def test_detect_german():
    code, _ = detect_language("Das ist wirklich ein sehr schlechtes Produkt und ich hasse es")
    assert code == "de"


def test_detect_japanese_by_script():
    code, script = detect_language("この商品は本当に素晴らしい")
    assert code == "ja"
    assert script == "cjk"


def test_detect_chinese_by_script():
    code, script = detect_language("这个产品非常好用")
    assert code == "zh"
    assert script == "cjk"


def test_detect_korean_by_script():
    code, script = detect_language("이 제품은 정말 최고입니다")
    assert code == "ko"
    assert script == "cjk"


def test_detect_arabic_by_script():
    code, script = detect_language("هذا المنتج رائع جدا")
    assert code == "ar"
    assert script == "arabic"


def test_detect_russian_stops_beat_default():
    code, script = detect_language("Этот продукт действительно ужасен и я его ненавижу")
    assert code == "ru"
    assert script == "cyrillic"


def test_detect_welsh_low_resource():
    code, _ = detect_language("Mae cynnyrch yn dda iawn ac rwyf wrth fy modd")
    assert code == "cy"


def test_gibberish_is_low_confidence():
    _code, script, confidence, _runner, _margin = detect_language_scored("xkcd qqq zzz www")
    assert script == "latin"
    assert confidence < 0.2


def test_emoji_only_is_unknown():
    code, script, confidence, _runner, _margin = detect_language_scored("😀😀🔥")
    assert code == "unknown"
    assert script == "emoji"
    assert confidence == 0.0


def test_empty_text():
    code, script, confidence, _runner, _margin = detect_language_scored("   ")
    assert code == "unknown"
    assert script == "none"
    assert confidence == 0.0


def test_normalize_lang_variants():
    assert normalize_lang("English") == "en"
    assert normalize_lang("eng") == "en"
    assert normalize_lang("Español") == "es"
    assert normalize_lang("spa") == "es"
    assert normalize_lang("pt-BR") == "pt"
    assert normalize_lang("Mandarin") == "zh"
    assert normalize_lang("") == "unknown"


def test_low_resource_flags():
    assert is_low_resource("cy")
    assert is_low_resource("sw")
    assert not is_low_resource("en")
    assert not is_low_resource("es")


def test_code_switching_latin():
    mixed = detect_code_switching("The product is good. El producto es bueno.", "latin")
    assert mixed == ["en", "es"]


def test_code_switching_single_language_quiet():
    assert detect_code_switching("This is fine and it is good.", "latin") == []


def test_code_switching_needs_two_sentences():
    assert detect_code_switching("The product is good todo genial", "latin") == []


def test_normalize_lang_iso639_3_codes():
    expected = {
        "jpn": "ja", "tur": "tr", "swe": "sv", "pol": "pl", "ces": "cs", "dan": "da",
        "nld": "nl", "rus": "ru", "zho": "zh", "kor": "ko", "ara": "ar", "heb": "he",
        "ell": "el", "tha": "th", "hin": "hi", "ben": "bn", "urd": "ur", "vie": "vi",
        "ind": "id", "mal": "ms", "ukr": "uk", "ron": "ro", "hun": "hu", "fin": "fi",
        "nor": "no", "slo": "sk", "slk": "sk", "slv": "sl", "lit": "lt", "lav": "lv",
        "est": "et", "sqi": "sq", "isl": "is", "cat": "ca", "glg": "gl", "cym": "cy",
        "gle": "ga", "eus": "eu", "afr": "af", "swa": "sw", "tgl": "tl", "hrv": "hr",
    }
    for code, language in expected.items():
        assert normalize_lang(code) == language, code
        assert normalize_lang(code.upper()) == language, code


def test_runner_up_ignores_zero_hit_languages():
    _code, _script, _confidence, runner_up, margin = detect_language_scored(
        "the and el los que para"
    )
    assert runner_up == "en"
    assert margin == 2.0


def test_no_runner_up_has_zero_margin():
    _code, _script, _confidence, runner_up, margin = detect_language_scored("the and")
    assert runner_up == ""
    assert margin == 0.0


def test_short_text_confidence_is_capped():
    code, _script, confidence, _runner, _margin = detect_language_scored("the and")
    assert code == "en"
    assert 0.2 <= confidence <= 0.85

    longer_confidence = detect_language_scored(
        "the and of to that is in you for it this with are"
    )[2]
    assert longer_confidence > confidence


def test_code_switching_across_scripts():
    assert detect_code_switching("यह अच्छा है। هذا منتج رائع.") == ["ar", "hi"]
    assert detect_code_switching("This is good مرحبا بالعالم") == ["ar", "en"]
    assert detect_code_switching("这个产品很好 This is good") == ["en", "zh"]
    assert detect_code_switching("Это хорошо. Αυτό είναι καλό.") == ["el", "ru"]


def test_code_switching_single_script_stays_quiet():
    assert detect_code_switching("Esto es lo peor que he comprado y no lo recomiendo") == []
    assert detect_code_switching("この商品は本当に素晴らしい") == []


def test_language_distribution_single_and_mixed():
    assert language_distribution("Esto es lo peor que he comprado y no lo recomiendo") == {"es": 1.0}
    mixed = language_distribution("The product is good. El producto es bueno.")
    assert mixed == {"en": 0.5, "es": 0.5}
    assert language_distribution("") == {}
