import re
from typing import Optional

LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
    "mai": "Maithili",
    "sat": "Santali",
    "ks": "Kashmiri",
    "sd": "Sindhi",
    "ne": "Nepali",
    "kok": "Konkani",
    "doi": "Dogri",
    "mni": "Manipuri",
    "bho": "Bhojpuri",
    "mag": "Magahi",
    "hne": "Chhattisgarhi",
    "raj": "Rajasthani",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
}

INDIAN_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "hi": [(0x0900, 0x097F)],
    "bn": [(0x0980, 0x09FF)],
    "ta": [(0x0B80, 0x0BFF)],
    "te": [(0x0C00, 0x0C7F)],
    "mr": [(0x0900, 0x097F)],
    "gu": [(0x0A80, 0x0AFF)],
    "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)],
    "pa": [(0x0A00, 0x0A7F)],
    "ur": [(0x0600, 0x06FF)],
    "or": [(0x0B00, 0x0B7F)],
    "as": [(0x0980, 0x09FF)],
}


def language_name(code: str) -> str:
    return LANGUAGE_MAP.get(code, f"Unknown ({code})")


class LanguageDetector:
    def __init__(self):
        self._detector = None

    def _init_detector(self):
        if self._detector is not None:
            return
        try:
            from langdetect import LangDetect
            self._detector = LangDetect()
        except ImportError:
            try:
                from langdetect import detect
                self._detector = detect
            except ImportError:
                self._detector = None

    def detect(self, text: str) -> str:
        if not text or not text.strip():
            return "en"
        text = text.strip()
        script_lang = self._detect_by_script(text)
        if script_lang and script_lang != "hi":
            return script_lang
        return self._detect_by_library(text) or script_lang or "en"

    def detect_name(self, text: str) -> str:
        code = self.detect(text)
        return language_name(code)

    def _detect_by_script(self, text: str) -> Optional[str]:
        for code, ranges in INDIAN_SCRIPT_RANGES.items():
            for cp in text:
                point = ord(cp)
                for start, end in ranges:
                    if start <= point <= end:
                        return code
        return None

    def _detect_by_library(self, text: str) -> str:
        try:
            from langdetect import detect
            return detect(text)
        except ImportError:
            pass
        try:
            from lingua import LanguageDetector as LinguaDetector
            from lingua import Language
            detector = LinguaDetector.from_languages(
                [Language.ENGLISH, Language.HINDI, Language.BENGALI,
                 Language.TAMIL, Language.TELUGU, Language.MARATHI,
                 Language.GUJARATI, Language.URDU]
            )
            result = detector.detect_language_of(text)
            if result:
                return result.iso_code_639_1.name.lower()
        except ImportError:
            pass
        return "en"


language_detector = LanguageDetector()
