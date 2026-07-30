# tests/india/test_script.py
from photon.india.script import Script, detect_script, script_mix


def test_detects_major_indic_scripts():
    assert detect_script("नमस्ते दुनिया") is Script.DEVANAGARI      # Hindi/Marathi
    assert detect_script("வணக்கம் உலகம்") is Script.TAMIL
    assert detect_script("నమస్కారం") is Script.TELUGU
    assert detect_script("নমস্কার") is Script.BENGALI
    assert detect_script("ನಮಸ್ಕಾರ") is Script.KANNADA
    assert detect_script("നമസ്കാരം") is Script.MALAYALAM
    assert detect_script("નમસ્તે") is Script.GUJARATI
    assert detect_script("ਸਤ ਸ੍ਰੀ ਅਕਾਲ") is Script.GURMUKHI
    assert detect_script("ନମସ୍କାର") is Script.ODIA


def test_detects_latin_and_empty():
    assert detect_script("hello world") is Script.LATIN
    assert detect_script("") is Script.UNKNOWN
    assert detect_script("12345 !!!") is Script.UNKNOWN  # no letters at all


def test_dominant_script_wins_in_code_mixed_text():
    # Hinglish reality: mixed Devanagari + Latin. Dominant script decides.
    mostly_hindi = "मुझे इस प्रोडक्ट के बारे में जानकारी चाहिए please"
    assert detect_script(mostly_hindi) is Script.DEVANAGARI
    mostly_english = "I need details about the प्रोडक्ट"
    assert detect_script(mostly_english) is Script.LATIN


def test_script_mix_reports_proportions():
    mix = script_mix("नमस्ते hello")
    assert mix[Script.DEVANAGARI] > 0
    assert mix[Script.LATIN] > 0
    assert abs(sum(mix.values()) - 1.0) < 1e-9


def test_script_mix_empty_is_empty():
    assert script_mix("") == {}


def test_is_indic_flag():
    assert Script.DEVANAGARI.is_indic is True
    assert Script.TAMIL.is_indic is True
    assert Script.LATIN.is_indic is False
    assert Script.UNKNOWN.is_indic is False
