"""Detector-level tests: language ID, normalization, code-switching."""

from __future__ import annotations

from linguacheck.languages import (
    detect_code_switching,
    detect_language,
    detect_language_scored,
    is_low_resource,
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
    code, script, confidence, _runner, _margin = detect_language_scored("xkcd qqq zzz www")
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
