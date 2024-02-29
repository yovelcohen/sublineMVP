from enum import Enum
from typing import NewType, Final, Literal

XMLString = str | bytes
SrtString = NewType('SrtString', str)
JsonStr = NewType('JsonStr', str)
VTTString = NewType('VTTString', str)

MALE: Final[str] = 'male'
FEMALE: Final[str] = 'female'
Gender = Literal[MALE, FEMALE]  # type: ignore


class AllowedSourceLanguages(str, Enum):
    """
    Supported languages for Transcribing Audio
    """

    de = "de"
    "German"

    en = "en"
    "Global English"

    en_au = "en_au"
    "Australian English"

    en_uk = "en_GB"
    "British English"

    en_us = "en_us"
    "English (US)"

    es = "es"
    "Spanish"

    fi = "fi"
    "Finnish"

    fr = "fr"
    "French"

    hi = "hi"
    "Hindi"

    it = "it"
    "Italian"

    ja = "ja"
    "Japanese"

    ko = "ko"
    "Korean"

    nl = "nl"
    "Dutch"

    pl = "pl"
    "Polish"

    pt = "pt"
    "Portuguese"

    ru = "ru"
    "Russian"

    tr = "tr"
    "Turkish"

    uk = "uk"
    "Ukrainian"

    vi = "vi"
    "Vietnamese"

    zh = "zh"
    "Chinese"

    he = "he"
    "Hebrew"

    HEBREW = he
